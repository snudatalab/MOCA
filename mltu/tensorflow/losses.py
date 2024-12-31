import tensorflow as tf


class CTCloss_ours(tf.keras.losses.Loss):
    """ CTCLoss objec for training the model"""
    def __init__(self, name: str = "CTCloss") -> None:
        super(CTCloss, self).__init__()
        self.name = name
        self.loss_fn = tf.keras.backend.ctc_batch_cost

    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight=None) -> tf.Tensor:
        """ Compute the training batch CTC loss value"""

        real_y_true, y_trues, y_weights = y_true.split("__")
        y_weights = y_weights.split("_")
        y_trues = y_trues.split("_")

        loss = 0
        for y_true, y_weights in zip(y_trues, y_weights):

            batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
            input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
            label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

            input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
            label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

            loss += y_weights * self.loss_fn(y_true, y_pred, input_length, label_length)

        return loss


class CTCLoss(nn.Module):
    """ CTC loss for PyTorch
    """

    def __init__(self, blank: int, reduction: str = "mean", zero_infinity: bool = False):
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