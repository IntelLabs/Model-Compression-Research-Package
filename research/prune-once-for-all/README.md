<!-- 
Apache v2 license
Copyright (C) 2021 Intel Corporation
SPDX-License-Identifier: Apache-2.0
 -->

# Prune Once for All: Train Your Own Sparse Pre-Trained Language Model

## Introduction
In [Prune Once for All: Sparse Pre-Trained Language Models](https://arxiv.org/abs/2111.05754) we present a new method for training sparse pre-trained Transformer language models by integrating weight pruning and model distillation. 
These sparse pre-trained models can be used to transfer learning for a wide range of tasks while maintaining their sparsity pattern.
We show how the compressed sparse pre-trained models we trained transfer their knowledge to five different downstream natural language tasks with minimal accuracy loss.
For example, with our sparse pre-trained BERT-Large fine-tuned on SQuADv1.1 and quantized to 8bit we achieve a compression ratio of 40X for the encoder with less than 1% accuracy loss.

In this example we will produce a 90% sparse pre-trained BERT-Base model using the Model-Compression-Research-Package.

## Datasets
Like in the paper, we will use English Wikipedia dataset.
We will download and process the dataset using `🤗/datasets`.

## Requirements
To run this example you will need to install Model-Compression-Research-Package as explained [here](../../README.md#Installation).

In addition you will need to install the requirements for running the language-modeling example.
```bash
pip install -r ../../examples/transformers/language-modeling/requirements.txt
```

## Teacher Preparation:
The teacher preparation step is necessary in case the initial model was not optimized on the exact dataset we want to use for pruning.
In this step we will train `bert-base-uncased` for 100k optimization steps with batch size 256 on English Wikipedia dataset processed in segment pair + NSP format, like in the original BERT paper.
To do that you can simply run the teacher preparation script:

```bash
./teacher_preparation.sh  # Check notes in the script for optimizing training
```

After running this script the teacher we prepared will be in `/tmp/bert-base-uncased-teacher-preparation`.

## Student Pruning
After preparing our teacher we will want to prune our model.
We prune the model using the same dataset from the teacher preparation step using the model we optimized to initialize both the student and the teacher.
In practice we repeat the same training from the previous step, only this time we will prune the student model while distilling the knowledge from the frozen teacher model.
To do that you can simply run the student pruning script:

```bash
./student_pruning.sh  # Check notes in the script for optimizing training
```

After running this script the pruned model will be in `/tmp/bert-base-uncased-sparse-90-pruneofa`.

That's it, we have our sparse pre-trained language model!

## Transfer Learning
Once we have our sparse pre-trained BERT-Base, we can utilize it for transfer learning for many natural language tasks such as question answering, text classification, token classification, etc.
All you need to do is lock the sparsity pattern of the model before fine-tuning it.
here is an example showing one way of doing it:

```python
import transformers
import model_compression_research as model_comp

model = transformers.AutoModelForQuestionAnswering.from_pretrained('Intel/bert-base-uncased-sparse-90-unstructured-pruneofa')

scheduler = mcr.pruning_scheduler_factory(model, '../../examples/transformers/question-answering/config/lock_config.json')

# Train your model...

scheduler.remove_pruning()
```

You can check our transformers [examples](../../examples/transformers) for more information on how to use `model_compression_research` with the HuggingFace `Trainer` and how to add quantization aware training.

## Our Results
Here we present the results from our paper:

| Model                         | Model Size | SQuADv1.1 (EM/F1) | MNLI-m (Acc) | MNLI-mm (Acc) | QQP (Acc/F1) | QNLI (Acc) | SST-2 (Acc) |
|-------------------------------|:----------:|:-----------------:|:------------:|:-------------:|:------------:|:----------:|:-----------:|
| [85% Sparse BERT-Base uncased](https://huggingface.co/Intel/bert-base-uncased-sparse-85-unstructured-pruneofa)  |   Medium   |    81.10/88.42    |     82.71    |     83.67     |  91.15/88.00 |    90.34   |    91.46    |
| [90% Sparse BERT-Base uncased](https://huggingface.co/Intel/bert-base-uncased-sparse-90-unstructured-pruneofa)  |   Medium   |    79.83/87.25    |     81.45    |     82.43     |  90.93/87.72 |    89.07   |    90.88    |
| [90% Sparse BERT-Large uncased](https://huggingface.co/Intel/bert-large-uncased-sparse-90-unstructured-pruneofa) |    Large   |    83.35/90.20    |     83.74    |     84.20     |  91.48/88.43 |    91.39   |    92.95    |
| [85% Sparse DistilBERT uncased](https://huggingface.co/Intel/distilbert-base-uncased-sparse-85-unstructured-pruneofa) |    Small   |    78.10/85.82    |     81.35    |     82.03     |  90.29/86.97 |    88.31   |    90.60    |
| [90% Sparse DistilBERT uncased](https://huggingface.co/Intel/distilbert-base-uncased-sparse-90-unstructured-pruneofa) |    Small   |    76.91/84.82    |     80.68    |     81.47     |  90.05/86.67 |    87.66   |    90.02    |

All the results are the mean of two seperate experiments with the same hyper-parameters and different seeds.
The sparse pre-trained models used to produce these results are available in [https://huggingface.co/Intel](https://huggingface.co/Intel)

## Citation
If you want to cite our paper, you can use the following:
```bibtex
@article{zafrir2021prune,
  title={Prune Once for All: Sparse Pre-Trained Language Models},
  author={Zafrir, Ofir and Larey, Ariel and Boudoukh, Guy and Shen, Haihao and Wasserblat, Moshe},
  journal={arXiv preprint arXiv:2111.05754},
  year={2021}
}
```