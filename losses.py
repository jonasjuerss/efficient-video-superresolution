import tensorflow as tf
from tensorflow.keras.losses import MeanAbsoluteError
from wavetf import WaveTFFactory


class Losses:
    def __init__(self, args):
        self.alpha = args.alpha
        self.wavelet_layer = WaveTFFactory().build(kernel_type = 'db2', dim = 2)

    @tf.function
    def train_step(self, model, x, ground_truth, teacher_output):
        with tf.GradientTape() as tape:
            student_output = model(x, training = True)
            student_loss = self.student_loss(ground_truth, student_output)
            distillation_loss = self.distillation_loss(teacher_output, student_output) * self.alpha
            loss_values = [student_loss, distillation_loss]
        return loss_values, tape.gradient(loss_values, model.trainable_variables)

    @tf.function
    def test_step(self, model, x, ground_truth, teacher_output):
        student_output = model(x, training = False)
        student_loss = self.student_loss(ground_truth, student_output)
        distillation_loss = self.distillation_loss(teacher_output, student_output) * self.alpha
        loss_values = [student_loss, distillation_loss]
        return loss_values, student_output

    @tf.function
    def student_loss(self, ground_truth, student_output):
        return MeanAbsoluteError()(ground_truth, student_output)

    @tf.function
    def distillation_loss(self, teacher_output, student_output):
        teacher_output = tf.reshape(teacher_output, (-1, 128, 128, 3))
        student_output = tf.reshape(student_output, (-1, 128, 128, 3))

        student_freqs1 = self.wavelet_layer(student_output)
        teacher_freqs1 = self.wavelet_layer(teacher_output)
        LL1_s = student_freqs1[:, :, :, :3]
        LL1_t = teacher_freqs1[:, :, :, :3]

        student_freqs2 = self.wavelet_layer(LL1_s)
        teacher_freqs2 = self.wavelet_layer(LL1_t)
        LL2_s = student_freqs2[:, :, :, :3]
        LL2_t = teacher_freqs2[:, :, :, :3]

        student_freqs3 = self.wavelet_layer(LL2_s)
        teacher_freqs3 = self.wavelet_layer(LL2_t)

        loss1 = MeanAbsoluteError()(student_freqs1[:, :, :, 3:], teacher_freqs1[:, :, :, 3:])
        loss2 = MeanAbsoluteError()(student_freqs2[:, :, :, 3:], teacher_freqs2[:, :, :, 3:])
        loss3 = MeanAbsoluteError()(student_freqs3[:, :, :, 3:], teacher_freqs3[:, :, :, 3:])

        return loss1 + loss2 + loss3