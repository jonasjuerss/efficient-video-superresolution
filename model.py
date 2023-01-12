import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, SeparableConv2D, ReLU, Add, RNN, AbstractRNNCell
from tensorflow.keras.activations import tanh
from tensorflow.keras.losses import MeanAbsoluteError


class CNNCell(AbstractRNNCell):
    def __init__(self, args, **kwargs):
        self.args = args
        super(CNNCell, self).__init__(**kwargs)

    def build(self, input_shape):
        self.model = CNNModel(self.args).get_model()
        self.built = True

    def call(self, inputs, states):
        # input: (batch, sequence_length, heightLR, widthLR, 3)

        # prev_output: (batch, sequence_length, heightHR, widthHR, 3)
        prev_output, _ = states

        # prev_output: (batch, sequence_length, heightLR, widthLR, 12)
        prev_output = tf.nn.space_to_depth(prev_output, self.block_size)

        x = tf.stack([prev_output, inputs], axis = -1)

        output = self.model(x)
        
        return output, output


class CNNModel:
    def __init__(self, args):
        # Sequence
        self.sequence_length = args.sequence_length

        # Residual blocks
        self.num_res_blocks = self.args.num_res_blocks

        # Depthwise separable 2D convolution
        self.filters = args.filters
        self.kernel_size = args.kernel_size
        self.strides = args.strides
        self.padding = args.padding

        # Pixel Shuffle
        self.block_size = args.block_size

        # Wavelet transformation
        self.wavelet_name = args.wavelet_name
        self.wavelet_level = args.wavelet_level

    def get_residual_block(self, input):
        x = SeparableConv2D(filters = self.filters, kernel_size = self.kernel_size, strides = self.strides, padding = self.padding)(input)
        x = ReLU()(x)
        x = Add()([input, x])
        return x

    def get_model(self):
        inputs = Input(shape = (self.sequence_length, None, None, 3 + 3 * 4))

        x = SeparableConv2D(filters = self.filters, kernel_size = self.kernel_size, strides = self.strides, padding = self.padding)(rnn)
        x = ReLU()(x) # TODO Skip more than one layer?

        for _ in self.num_res_blocks:
            x = self.get_residual_block(x)

        x = tf.nn.depth_to_space(input = x, block_size = self.block_size)
        outputs = tanh(x)

        model = Model(inputs = inputs, outputs = outputs)
        
        return model


class SmallModel:
    def __init__(self, args):
        self.args = args
        self.sequence_length = args.sequence_length
        self.wave_name = args.wave_name

    def get_model(self):
        inputs = Input(shape = (self.sequence_length, None, None, 3 + 3 * 4))
        rnn = RNN(CNNCell(self.args))(inputs)
        
        model = Model(inputs = inputs, outputs = rnn)

        return model

    def distillation_loss(self, student_output, teacher_output, wave_layer):
        student_freqs1 = wave_layer(student_output)
        teacher_freqs1 = wave_layer(teacher_output)
        LL1_s = student_freqs1[:, :, :, 0]
        LL1_t = teacher_freqs1[:, :, :, 0]

        student_freqs2 = wave_layer(LL1_s)
        teacher_freqs2 = wave_layer(LL1_t)
        LL2_s = student_freqs2[:, :, :, 0]
        LL2_t = teacher_freqs2[:, :, :, 0]

        student_freqs3 = wave_layer(LL2_s)
        teacher_freqs3 = wave_layer(LL2_t)

        loss1 = MeanAbsoluteError(student_freqs1[:, :, :, 1:], teacher_freqs1[:, :, :, 1:])
        loss2 = MeanAbsoluteError(student_freqs2[:, :, :, 1:], teacher_freqs2[:, :, :, 1:])
        loss3 = MeanAbsoluteError(student_freqs3[:, :, :, 1:], teacher_freqs3[:, :, :, 1:])

        return loss1 + loss2 + loss3