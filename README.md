<!-- 
Apache v2 license
Copyright (C) 2021 Intel Corporation
SPDX-License-Identifier: Apache-2.0
 -->
# Model Compression Research Package
This package was developed to enable scalable, reusable and reproducable research of weight pruning, quantization and distillation methods with ease.

## Installation
To install the library clone the repository and install using `pip`
``` bash
git clone https://github.com/IntelLabs/Model-Compression-Research-Package
cd Model-Compression-Research-Package
pip install [-e] .
```
Add `-e` flag to install an editable version of the library.

## Quick Tour
This package contains implementations of several weight pruning methods, knowledge distillation and quantization-aware training.
Here we will show how to easily use those implementations with your existing model implementation and training loop.
It is also possible to combine several methods together in the same training process.
Please refer to the packages [examples](examples).

### Weight Pruning
Weight pruning is a method to induce zeros in a models weight while training.
There are several methods to prune a model and it is a widely explored research field.

To list the existing weight pruning implemtations in the package use `model_compression_research.list_methods()`.
For example, applying unstructured magnitude pruning while training your model can be done with a few single lines of code

```python
from model_compression_research import IterativePruningConfig, IterativePruningScheduler

training_args = get_training_args()
model = get_model()
dataloader = get_dataloader()
criterion = get_criterion()

# Initialize a pruning configuration and a scheduler and apply it on the model
pruning_config = IterativePruningConfig(
    pruning_fn="unstructured_magnitude",
    pruning_fn_default_kwargs={"target_sparsity": 0.9}
)
pruning_scheduler = IterativePruningScheduler(model, pruning_config)

# Initialize optimizer after initializing the pruning scheduler
optimizer = get_optimizer()

# Training loop
for e in range(training_args.epochs):
    for batch in dataloader:
        inputs, labels = 
        model.train()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # Call pruning scheduler step
        pruning_schduler.step()
        optimizer.zero_grad()

# At the end of training rmeove the pruning parts and get the resulted pruned model
pruning_scheduler.remove_pruning()
```

For using knowledge distillation with [`HuggingFace/transformers`](https://github.com/huggingface/transformers) dedicated transformers [`Trainer`](https://huggingface.co/transformers/main_classes/trainer.html) see the implementation of `HFTrainerPruningCallback` in [`api_utils.py`](model_compression_research/api_utils.py).

### Knowledge Distillation
Model distillation is a method to distill the knowledge learned by a teacher to a smaller student model.
A method to do that is to compute the difference between the student's and teacher's output distribution using KL divergence.
In this package you can find a simple implementation that does just that.

Assuming that your teacher and student models' outputs are of the same dimension, you can use the implementation in this package as follows:
```python
from model_compression_research import TeacherWrapper, DistillationModelWrapper

training_args = get_training_args()
teacher = get_teacher_trained_model()
student = get_student_model()
dataloader = get_dataloader()
criterion = get_criterion()

# Wrap teacher model with TeacherWrapper and set loss scaling factor and temperature
teacher = TeacherWrapper(teacher, ce_alpha=0.5, ce_temperature=2.0)
# Initialize the distillation model with the student and teacher
distillation_model = DistillationModelWrapper(student, teacher, alpha_student=0.5)

optimizer = get_optimizer()

# Training loop
for e in range(training_args.epochs):
    for batch in dataloader:
        inputs, labels = batch
        distillation_model.train()
        # Calculate student loss w.r.t labels as you usually do
        student_outputs = distillation_model(inputs)
        loss_wrt_labels = criterion(student_outputs, labels)
        # Add knowledge distillation term
        loss = distillation_model.compute_loss(loss_wrt_labels, student_outputs)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

For using knowledge distillation with [`HuggingFace/transformers`](https://github.com/huggingface/transformers) see the implementation of `HFTeacherWrapper` and `hf_add_teacher_to_student` in [`api_utils.py`](model_compression_research/api_utils.py).

### Quantization-Aware Training
Quantization-Aware Training is a method for training models that will be later quantized at the inference stage, as opposed to other post-training quantization methods where models are trained without any adaptation to the error caused by model quantization.

A similar quantization-aware training method to the one introduced in [Q8BERT: Quantized 8Bit BERT](https://arxiv.org/abs/1910.06188) generelized to custom models is implemented in this package:

```python
from model_compression_research import QuantizerConfig, convert_model_for_qat

training_args = get_training_args()
model = get_model()
dataloader = get_dataloader()
criterion = get_criterion()

# Initialize quantizer configuration
qat_config = QuantizerConfig()
# Convert model to quantization-aware training model
qat_model = convert_model_for_qat(model, qat_config)

optimizer = get_optimizer()

# Training loop
for e in range(training_args.epochs):
    for batch in dataloader:
        inputs, labels = 
        model.train()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

## Papers Implemented in Model Compression Research Package
Methods from the following papers were implemented in this package and are ready for use:
* [To prune, or not to prune: exploring the efficacy of pruning for model compression](https://arxiv.org/abs/1710.01878)
* [Discovering Neural Wirings](https://arxiv.org/abs/1906.00586)
* [Q8BERT: Quantized 8Bit BERT](https://arxiv.org/abs/1910.06188)
* [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
* [Prune Once for All: Sparse Pre-Trained Language Models](https://arxiv.org/abs/2111.05754)

## Citation
If you want to cite our paper and library, you can use the following:
```bibtex
@article{zafrir2021prune,
  title={Prune Once for All: Sparse Pre-Trained Language Models},
  author={Zafrir, Ofir and Larey, Ariel and Boudoukh, Guy and Shen, Haihao and Wasserblat, Moshe},
  journal={arXiv preprint arXiv:2111.05754},
  year={2021}
}
```