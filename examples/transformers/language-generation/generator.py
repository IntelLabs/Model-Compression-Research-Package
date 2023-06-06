# Apache v2 license
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from dataclasses import dataclass
from typing import Optional, List
import os
import warnings

import transformers
import tqdm
import torch
from torch import nn
import intel_extension_for_pytorch as ipex

try:
    from . import utils
except:
    import utils

# Enable dynamic shape patch
torch._C._jit_set_texpr_fuser_enabled(False)


logger = logging.getLogger(__name__)


# Default sequence lengths to warmup generation model
DEFAULT_PREFIX_LIST = [1, 8, 16, 32, 64, 128, 256]


SEPERATOR = '-' * 72 + '\n'


@dataclass
class GenerationOutput:
    input_text: Optional[str] = None
    input_tokens: Optional[List] = None
    generated_tokens: List = None
    generated_text: Optional[str] = None
    time_token: Optional[List[float]] = None
    time_first_token: Optional[float] = None
    total_time: Optional[float] = None
    FRAME: str = SEPERATOR

    @property
    def mean_time_token(self):
        if self.time_token is not None and len(self.time_token) > 0:
            return sum(self.time_token) / len(self.time_token)

    @property
    def generated_tokens_num(self):
        return len(self.generated_tokens)

    @property
    def input_tokens_num(self):
        return len(self.input_tokens)

    @property
    def output_text(self):
        return self.input_text + self.generated_text

    @property
    def long_metadata_string(self):
        s = self.FRAME
        s += f'Input tokens number: {self.input_tokens_num}\n'
        s += f'Generated tokens number: {self.generated_tokens_num}\n'
        if self.total_time is not None:
            s += f'Total generation time: {self.total_time / 1000:.3f} seconds\n'
        if self.time_first_token is not None:
            s += f'First token generation time: {self.time_first_token:.3f} miliseconds\n'
        if self.mean_time_token is not None:
            s += f'Mean token generation time: {self.mean_time_token:.3f} miliseconds (excluding the first token)\n'
        s += self.FRAME
        return s

    @property
    def metadata_string(self):
        s = ''
        s += f'input_tokens_num={self.input_tokens_num}'
        s += f', generated_tokens_num={self.generated_tokens_num}'
        if self.total_time is not None:
            s += f', gen_total_time={self.total_time / 1000:.3f}s'
        return s


class GenerationModelWrapper(nn.Module, transformers.GenerationMixin):
    JIT_MODEL_NAME = 'frozen_model.pt'
    WARMUP_STEPS = 3

    def __init__(self,
                 model_path,
                 *,
                 generation_state=None,
                 prefix_sequence_lengths=None,
                 config_name_or_path=None,
                 generation_config_name_or_path=None,
                 tokenizer=None,
                 tokenizer_name_or_path=None,
                 tokenizer_kwargs=None,
                 main_input_name='input_ids',
                 device='cpu',
                 trust_remote_code=False,
                 verbose=False,
                 ):
        f"""
        prefix_sequence_lengths are the lengths that will be used to warmup the model.
        Also, every input will be padded to the next sequence lenght that can contain it.
        Each prefix length will increase the DRAM memory required to run the model,
        but, a finer granularity of prefix lengths might yield better latency results.
        Our experiments show that multiples of 64 work quite well.
        If prefix_sequence_lengths are not provided, the model will fall back to the default
        prefix lengths: {DEFAULT_PREFIX_LIST}
        """
        super().__init__()
        self.verbose = verbose
        self.model_mode = 'jit'
        logger.debug('Loading model')
        try:
            self.model = torch.jit.load(
                os.path.join(model_path, self.JIT_MODEL_NAME))
        except ValueError:
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=trust_remote_code)
            self.model_mode = 'hf'
        logger.debug(f'Model mode: {self.model_mode}')
        logger.debug('Loading config')
        self.config = transformers.AutoConfig.from_pretrained(
            config_name_or_path if config_name_or_path is not None else model_path, trust_remote_code=trust_remote_code)
        logger.debug('Loading generation config')
        try:
            self.generation_config = transformers.GenerationConfig.from_pretrained(
                generation_config_name_or_path if generation_config_name_or_path is not None else model_path, trust_remote_code=trust_remote_code)
        except OSError:
            self.generation_config = transformers.GenerationConfig.from_model_config(
                self.config)
        logger.debug('Loading tokenizer')
        tokenizer_kwargs = {} if tokenizer_kwargs is None else tokenizer_kwargs
        tokenizer_kwargs = {
            'padding_side': 'left',
            'model_input_names': ['input_ids', 'attention_mask']
        } | tokenizer_kwargs
        self.tokenizer = tokenizer
        if self.tokenizer is None:
            try:
                self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                    tokenizer_name_or_path if tokenizer_name_or_path is not None else model_path,
                    **tokenizer_kwargs
                )
            except OSError:
                logger.warning("Tokenizer wasn't loaded")
                self.tokenizer = None
        else:
            for k, v in tokenizer_kwargs.items():
                setattr(self.tokenizer, k, v)
        if self.tokenizer is not None and self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = '<pad>'
            logger.debug(
                f'Setting pad token to "{self.tokenizer.pad_token}"')
        # Load KV cache
        logger.debug('Loading cache config')
        # TODO: Use config to take the correct main input name
        self.main_input_name = main_input_name
        self.generation_state = {
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        self.update_generation_state(generation_state)
        self.device = torch.device(device) if type(device) == str else device
        self.prefix_sequence_lengths = prefix_sequence_lengths if prefix_sequence_lengths is not None else DEFAULT_PREFIX_LIST
        try:
            self.streamer = transformers.TextStreamer(
                self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        except AttributeError:
            self.streamer = None
        self._reorder_cache_fn = transformers.dynamic_module_utils.get_class_from_dynamic_module(
            self.config.auto_map['AutoModelForCausalLM'], model_path)
        if self.model_mode == 'jit':
            self.cache_config = utils.CacheConfig.from_pretrained(model_path)
            self.warmup(self.prefix_sequence_lengths)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, use_cache=None, **kwargs):
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)
        if past_key_values:
            input_ids = input_ids[:, -1:]
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'past_key_values': past_key_values,
        }

    @torch.inference_mode()
    def forward(self, input_ids, attention_mask=None, past_key_values=None, **kwargs):
        if self.model_mode == 'hf':
            return self.model(input_ids=input_ids, attention_mask=attention_mask, past_key_values=past_key_values, **kwargs)
        # elif self.model_mode == 'jit':
        if len(kwargs) > 0 and self.verbose:
            warnings.warn(
                f'The following model input arguments are ignored: {kwargs}')
        if attention_mask is None:
            attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()
        model_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'past_key_values': past_key_values
        }
        with torch.cpu.amp.autocast(enabled=True):
            past_key_values = self._maybe_handle_long_history(model_inputs)
            if past_key_values is None:
                past_key_values = self.dummy_past_key_values(
                    0, input_ids.size(0))
            if input_ids.size(-1) + past_key_values[0][1].size(-2) != attention_mask.size(-1):
                model_inputs['input_ids'] = model_inputs['input_ids'][:,
                                                                      past_key_values[0][0].shape[-2]:]
            model_inputs['past_key_values'] = past_key_values
            out = self.model(**model_inputs)
        return transformers.modeling_outputs.CausalLMOutputWithPast(logits=out[0], past_key_values=out[1])

    def dummy_input(self, batch_size, sequence_length, past_kv_length):
        return utils.generate_sample_config(
            batch_size,
            sequence_length,
            past_kv_length,
            cache_config=self.cache_config,
        )

    def dummy_past_key_values(self, length, batch_size=1):
        return self.cache_config.generate_past_key_values(length, batch_size)

    @torch.inference_mode()
    def warmup(self, sequence_lengths, batch_size=1):
        sequence_lengths = set(sequence_lengths)
        logger.debug(
            f'Warming up for sequence lengths: {list(sequence_lengths)}')
        for length in tqdm.tqdm(sorted(list(sequence_lengths), reverse=True), desc='Warmup', disable=not self.verbose):
            logger.debug(f'Warming up sequence length {length}')
            for _ in range(self.WARMUP_STEPS):
                self(**self.dummy_input(batch_size, length, 0))

    def get_input_patitioning(self, input_length):
        sequences = []
        done = False
        while not done:
            for length in self.prefix_sequence_lengths:
                if input_length <= length:
                    done = True
                    break
            input_length -= length
            sequences.append(length)
        return sequences

    def pad_tokenized(self, tokenized, tokenizer=None, return_sequences=False, **tokenizer_kwargs):
        if tokenizer is None:
            tokenizer = self.tokenizer
        input_length = len(tokenized.input_ids[0])
        logger.debug(f'Input tokens count {input_length}')
        sequences = self.get_input_patitioning(input_length)
        padded_length = sum(sequences)
        sequences = []
        logger.debug(f'Padding length {padded_length}')
        out = tokenizer.pad(tokenized, padding='max_length',
                            max_length=padded_length, **tokenizer_kwargs)
        if return_sequences:
            out = (out, sequences)
        return out

    def tokenize(self, text, return_sequences=False):
        tokenized = self.tokenizer(
            [text], return_attention_mask=False, return_token_type_ids=False)
        return self.pad_tokenized(tokenized, return_sequences=return_sequences, return_tensors='pt')

    def decode(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def update_generation_state(self, generation_state):
        if generation_state is not None:
            self.generation_state |= generation_state

    def can_generate(self):
        return True

    def _maybe_handle_long_history(self, model_input, lengths_list=None):
        if (past_key_values := model_input.get('past_key_values', None)) is None:
            if lengths_list is None:
                lengths_list = self.get_input_patitioning(
                    model_input['input_ids'].size(-1))
            if len(lengths_list) > 1:
                cur_length = 0
                logger.debug(f"Processing long history {lengths_list}")
                for l in lengths_list[:-1]:
                    cur_length += l
                    tmp_input = {k: v[:, (cur_length - l if k == self.main_input_name else 0):cur_length]
                                 for k, v in model_input.items() if v is not None}
                    past_key_values = self(
                        **tmp_input, past_key_values=past_key_values).past_key_values
        return past_key_values

    @torch.inference_mode()
    def generate(self, text, *, print_generated=False, stream=True, **generation_kwargs):
        times = []
        generated_tokens = []
        decoded_generated_tokens = []
        num_beams = (self.generation_state |
                     generation_kwargs).get('num_beams', 1)
        with torch.cpu.amp.autocast(enabled=True):
            with utils.Timer() as global_timer:
                # model_input, lengths_list = self.tokenize(text, return_sequences=True)
                model_input = self.tokenize(text)
                input_length = model_input['input_ids'].size(1)
                logger.debug(
                    f'Calling generate with length {input_length}')
                if self.model_mode == 'jit':
                    generated_tokens = list(
                        super().generate(
                            **model_input,
                            **(self.generation_state | generation_kwargs),
                            streamer=self.streamer if print_generated and stream and num_beams == 1 else None,
                        )
                    )[0][input_length:].tolist()
                elif self.model_mode == 'hf':
                    generated_tokens = list(
                        self.model.generate(
                            **model_input,
                            **(self.generation_state | generation_kwargs),
                            streamer=self.streamer if print_generated and stream and num_beams == 1 else None,
                        )
                    )[0][input_length:].tolist()
                else:
                    raise RuntimeError(
                        f'Received model mode which is not supported: {self.model_mode}')
                decoded_generated_tokens = self.decode(generated_tokens)
        out = GenerationOutput(
            input_text=text,
            input_tokens=model_input['input_ids'][0].tolist(),
            generated_tokens=generated_tokens,
            generated_text=''.join(decoded_generated_tokens),
            total_time=global_timer.time,
            time_first_token=times[0] if times else None,
            time_token=times[1:] if times else None,
        )
        if print_generated and (not stream or num_beams > 1):
            print(out.generated_text, flush=True)
        return out

    def _reorder_cache(self, *args, **kwargs):
        return self._reorder_cache_fn(*args, **kwargs)
