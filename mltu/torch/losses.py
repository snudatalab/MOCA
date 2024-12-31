import torch
import torch.nn as nn
import time

import numpy as np

class CTCLoss(nn.Module):
    """ CTC loss for PyTorch
    """
    def __init__(self, blank: int, reduction: str="mean", zero_infinity: bool=False):
        """ CTC loss for PyTorch

        Args:
            blank: Index of the blank label
        """
        super(CTCLoss, self).__init__()
        self.ctc_loss = nn.CTCLoss(blank=blank, reduction=reduction, zero_infinity=zero_infinity)
        self.blank = blank

    def forward(self, output, target):
        """
        Args:
            output: Tensor of shape (batch_size, num_classes, sequence_length)
            target: Tensor of shape (batch_size, sequence_length)
            
        Returns:
            loss: Scalar
        """
        # Remove padding and blank tokens from target
        target_lengths = torch.sum(target != self.blank, dim=1)
        using_dtype = torch.int32 if max(target_lengths) <= 256 else torch.int64
        device = output.device

        target_unpadded = target[target != self.blank].view(-1).to(using_dtype)

        output = output.permute(1, 0, 2)  # (sequence_length, batch_size, num_classes)
        output_lengths = torch.full(size=(output.size(1),), fill_value=output.size(0), dtype=using_dtype).to(device)

        loss = self.ctc_loss(output, target_unpadded, output_lengths, target_lengths.to(using_dtype))

        return loss

def score_to_weight(x, temperature=1):
    x = np.array([float(x_) for x_ in x])
    e_x = np.exp(x / temperature - np.max(x / temperature))
    return e_x / e_x.sum(axis=0)

def score_to_conf(x, temperature=1):
    if isinstance(x, float):
        x = [x]  # Convert single float to list
    x = np.array([float(x_) for x_ in x])
    e_x = np.exp(x / temperature)
    return e_x

# class beam_CTCLoss(nn.Module):
#     def __init__(self, blank: int, reduction: str = "mean", zero_infinity: bool = False):
#         super(beam_CTCLoss, self).__init__()
#         self.ctc_loss = nn.CTCLoss(blank=blank, reduction=reduction, zero_infinity=zero_infinity)
#         self.blank = blank
#
#     def compute_loss(self, output, target):
#         target_lengths = torch.sum(target != self.blank, dim=1)
#         using_dtype = torch.int32 if max(target_lengths) <= 256 else torch.int64
#         device = output.device
#
#         target_unpadded = target[target != self.blank].view(-1).to(using_dtype)
#
#         output = output.permute(1, 0, 2)  # (sequence_length, batch_size, num_classes)
#         output_lengths = torch.full(size=(output.size(1),), fill_value=output.size(0), dtype=using_dtype).to(device)
#
#         loss = self.ctc_loss(output, target_unpadded, output_lengths, target_lengths.to(using_dtype))
#
#         return loss
#
#     def forward(self, output, target):
#         print(target)
#         time.sleep(10)
#         # if len(target.split("__"))==1:
#         #     loss = self.compute_loss(output, target)
#         # else:
#         #     targets, scores = target.split("__")
#         #     targets = targets.split("_")
#         #     scores = scores.split("_")
#         #
#         #     loss = 0
#         #     for target, score in zip(targets, scores):
#         #         loss += -1/float(score) * self.compute_loss(target)
#
#         return loss
