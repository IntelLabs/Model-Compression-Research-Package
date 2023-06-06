# Apache v2 license
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Demo for running generation using quantized models with IPEX
"""
import argparse
import os
import logging
import warnings
import sys

import torch

import utils
import generator


# Global verbose variable
VERBOSE = False


DEFAULT_GENERATION_STATE = {
    'max_new_tokens': 128,
    'do_sample': True,
    'temperature': 0.1,
    'num_beams': 1,
    'top_p': 1.0,
    'top_k': 0,
    'repetition_penalty': 1.1,
}


ALPACA_PROMPT = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n" \
    "### Instruction:\n" \
    "{input}\n\n" \
    "### Response:\n"

VICUNA_PROMPT = "A chat between a curious user and an artificial intelligence assistant. " \
    "The assistant gives helpful, detailed, and polite answers to the user's questions. " \
    "USER: {input} ASSISTANT:"


PROMPTS = {
    'none': "{input}",
    'alpaca': ALPACA_PROMPT,
    'vicuna': VICUNA_PROMPT,
}


FRAME = '=' * 96 + '\n'


# Logger
logging.captureWarnings(True)
logger = logging.getLogger(__name__)


class Prompt:
    def __init__(self, prompt_name_or_path):
        self.prompt = PROMPTS.get(prompt_name_or_path, None)
        if self.prompt is None:
            raise NotImplementedError("Custom prompts are not implemented")

    def generate(self, inputs):
        return self.prompt.format(input=inputs)


def dict_none_filter(d):
    return {k: v for k, v in filter(lambda kv: kv[1] is not None, d.items())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str,
                        help='Path to torchscript model')
    parser.add_argument('-w', '--warmup_steps', type=int, default=5,
                        help='Number of warmup samplesto use for benchmarking, should be more than 2')
    parser.add_argument('-v', '--verbose', default=False, action=argparse.BooleanOptionalAction,
                        help='Set verbose to see more detailed logging')
    parser.add_argument('-p', '--profile', default=False,
                        action='store_true', help='Activate PyTorch profiler')
    parser.add_argument('-d', '--output_dir', default=None, type=str, nargs='?', const='',
                        help='Path to output directory to store results and profiler results, if left without an argument will save results <model_directory>/generation_results')
    parser.add_argument('--overwrite_output_dir', default=False,
                        action='store_true', help='If output_dir exists, overwrite its contents')
    parser.add_argument('--config_name_or_path', type=str, default=None,
                        help='Transformers model config name or path, must be provided for KV cache benchmarking')
    parser.add_argument('--generation_config_name_or_path', type=str, default=None,
                        help='Transformers generation config name or path, must be provided for KV cache benchmarking')
    parser.add_argument('--dynamic_shape', default=True, action=argparse.BooleanOptionalAction,
                        help='Cancel the option to store dynamic shape kernels')
    parser.add_argument('--debug', default=False,
                        action='store_true', help='Enable debug messages')
    parser.add_argument('--tokenizer_name_or_path', default=None, type=str,
                        help='Huggingface tokenizer name or path to pretrained tokenizer')
    parser.add_argument('--warmup_lengths', type=int, default=None,
                        nargs='*', help='Manually decide which warmup length to use')
    parser.add_argument('--prompt_name_or_path', type=str, default='none',
                        help='Name of predefined prompt or path to a txt file containing a prompt')
    parser.add_argument('--send_input_on_eof', default=False, action='store_true',
                        help='Send input on user EOF instead of newline (Enter)')
    parser.add_argument('--trust_remote_code', default=False,
                        action='store_true', help='HF trust remote code')

    generation_parser = argparse.ArgumentParser(
        prog='Generation demo', add_help=False)
    generation_parser.add_argument('--temperature', type=float,
                                   default=None, help='Generation temperature')
    generation_parser.add_argument('--do_sample', default=None, action=argparse.BooleanOptionalAction,
                                   help='Enable multinomial sampling')
    generation_parser.add_argument(
        '--max_new_tokens', type=int, default=None, help='The number of max new tokens to generate')
    generation_parser.add_argument(
        '--num_beams', default=None, type=int, help='Perform beam search with #num_beams')
    generation_parser.add_argument('--top_p', type=float, default=None,
                                   help='If < 1.0, only keep the top tokens with cumulative probability >= top_p')
    generation_parser.add_argument('--top_k', type=int, default=None,
                                   help='The size of the candidate set that is used to re-rank for contrastive search')
    generation_parser.add_argument('--repetition_penalty', type=float, default=None,
                                   help='The parameter for repetition penalty. 1.0 means no penalty. See [this paper](https://arxiv.org/pdf/1909.05858.pdf) for more details')

    args, gen_string = parser.parse_known_args()
    generation_args = generation_parser.parse_args(gen_string)

    # Set global verbosability
    global VERBOSE
    VERBOSE = args.verbose

    # Setup logger
    logging.basicConfig(
        format='%(asctime)s %(levelname)s: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    logger_level = logging.WARNING
    if args.debug:
        VERBOSE = True
        logger_level = logging.DEBUG
    elif VERBOSE:
        logger_level = logging.INFO
    logger.setLevel(logger_level)
    logging.root.setLevel(logger_level)
    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

    # Check arguments
    if args.output_dir == '':
        args.output_dir = os.path.join(
            os.path.dirname(args.model_path), 'generation_results')
        logger.debug(f'Set output_dir to {args.output_dir}')

    # Enable multiple length kernels
    if args.dynamic_shape:
        torch._C._jit_set_texpr_fuser_enabled(False)
        logger.debug('Enabled dynamic shape')

    # Setup output directory
    if args.output_dir is not None:
        os.makedirs(
            args.output_dir, exist_ok=args.overwrite_output_dir)
        logger.debug('Created output directory')

    generation_state = DEFAULT_GENERATION_STATE | dict_none_filter(
        vars(generation_args).copy())
    # Load model and tokenizer
    with utils.Timer('Load model', units='seconds', verbose=VERBOSE):
        model = generator.GenerationModelWrapper(
            args.model_path,
            config_name_or_path=args.config_name_or_path,
            generation_config_name_or_path=args.generation_config_name_or_path,
            tokenizer_name_or_path=args.tokenizer_name_or_path,
            generation_state=generation_state,
            prefix_sequence_lengths=args.warmup_lengths,
            trust_remote_code=args.trust_remote_code,
            verbose=VERBOSE,
        )
    model.eval()

    prompt = Prompt(args.prompt_name_or_path)

    # Add exit and usage option to the generation parser
    generation_parser.add_argument(
        '--exit', default=False, action='store_true', help='If given, exit the demo')
    generation_parser.add_argument(
        '--usage', default=False, action='store_true', help='Prints this help instructions')

    # Generate
    logger.info('Start generating')
    try:
        while True:
            with warnings.catch_warnings():
                if not args.debug:
                    warnings.simplefilter('ignore')
                while True:
                    print(f'{FRAME}\nCurrent generation config: {model.generation_state}\nEnter prompt or leave empty to enter settings followed by {"EOF (ctrl+D)" if args.send_input_on_eof else "Enter"}:\n')
                    # Read from stdin until receiving an EOF, this will allow user to input more complex prompts
                    user_input = sys.stdin.read() if args.send_input_on_eof else input()
                    if not user_input:
                        break
                    user_prompt = prompt.generate(user_input)
                    logger.debug(f"Generating with prompt:\n{user_prompt}")
                    gen_output = model.generate(
                        user_prompt, print_generated=True)
                    print()
                    if VERBOSE:
                        print(gen_output.long_metadata_string)
            print(FRAME)
            while True:
                generation_parser.print_usage()
                gen_string = input(f'Enter new generation arguments:\n')
                if len(gen_string) > 0:
                    generation_args, leftover = generation_parser.parse_known_args(
                        gen_string.split(' '))
                    if generation_args.usage:
                        generation_parser.print_help()
                        continue
                    if len(leftover) > 0:
                        print(
                            f'Failed to get new generation config, Got unknown arguments: {" ".join(leftover)}')
                        continue
                    gen_config = dict_none_filter(vars(generation_args).copy())
                    del gen_config['exit']
                    del gen_config['usage']
                    model.update_generation_state(gen_config)
                break
            if hasattr(generation_args, 'exit') and generation_args.exit:
                break
    except KeyboardInterrupt:
        pass

    logger.info('Done')


if __name__ == '__main__':
    main()
