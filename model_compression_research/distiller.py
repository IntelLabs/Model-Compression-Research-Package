# Apache v2 license
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Knowledge Distillation Helper class
"""
# import abc

import torch
from torch import nn
from torch.nn import functional as F


class TeacherWrapper(nn.Module):
    """Model distillation teacher wrapper class"""

    def __init__(self, teacher, *, ce_alpha=0., ce_temperature=1., convert_parameters=True, keep_gradients=False):
        super().__init__()
        self.teacher = teacher
        self.keep_gradients = keep_gradients
        self.ce_alpha = ce_alpha
        self.ce_temperature = ce_temperature
        if convert_parameters:
            self.convert_parameters_to_buffers()
        self._output = None

    def forward(self, *args, **kwargs):
        """Compute teacher forward and return teacher output"""
        self.teacher.eval()
        # In case output wasn't delted yet, delete output to make room for new output
        if hasattr(self, '_output'):
            del self._output
        with torch.set_grad_enabled(self.keep_gradients):
            self._output = self.teacher(*args, **kwargs)
        return self._output

    def convert_parameters_to_buffers(self):
        """Convert teacher module parameters to module buffers"""
        for m in self.teacher.modules():
            for n, p in list(m.named_parameters(recurse=False)):
                delattr(m, n)
                m.register_buffer(n, p.data)

    def compute_cross_entropy_loss(self, student_outputs, teacher_outputs):
        """Compute cross entropy loss"""
        return F.kl_div(
            input=F.log_softmax(student_outputs / self.ce_temperature, dim=-1),
            target=F.softmax(teacher_outputs / self.ce_temperature, dim=-1),
            reduction="batchmean"
        ) * (self.ce_temperature ** 2)

    def compute_distill_loss_callback(self, student_outputs, teacher_outputs=None):
        """Compute the distillation loss w.r.t teacher"""
        return self.compute_cross_entropy_loss(student_outputs, teacher_outputs) * self.ce_alpha

    def compute_distill_loss(self, student_outputs, teacher_outputs=None):
        """Compute the distillation loss w.r.t teacher scaled with teacher alpha"""
        teacher_outputs = self.get_teachers_outputs(teacher_outputs)
        distill_loss = self.compute_distill_loss_callback(
            student_outputs, teacher_outputs)
        # After calculation of the distillation loss we delete the output to conserve memory
        del self._output
        return distill_loss

    def get_teachers_outputs(self, teacher_outputs=None):
        """Get teacher's cached outputs"""
        if teacher_outputs is None:
            teacher_outputs = self._output
        return teacher_outputs


class DistillationModelWrapper(nn.Module):
    """
    Model distillation wrapper combining student and teachers to a single model that 
    outputs the knowledge distillation loss in the forward pass
    """

    def __init__(self, student, teachers, *, alpha_student=1., **_):
        super().__init__()
        self.student = student
        teachers = teachers if isinstance(teachers, list) else [teachers]
        for teacher in teachers:
            if not isinstance(teacher, TeacherWrapper):
                raise RuntimeError(
                    "Recieved a teacher not wrapped with TeacherWrapper class")
        self.teachers = nn.ModuleList(teachers)
        self.alpha_student = alpha_student

    def forward(self, *args, **kwargs):
        if self.training:
            for teacher in self.teachers:
                teacher(*args, **kwargs)
        return self.student(*args, **kwargs)

    def compute_loss(self, student_loss, student_outputs):
        """Compute combined loss of student with teachers"""
        loss = student_loss
        if self.training:
            loss *= self.alpha_student
            for teacher in self.teachers:
                loss += teacher.compute_distill_loss(student_outputs)
        return loss
