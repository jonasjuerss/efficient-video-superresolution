import tensorflow as tf
from keras.layers import Conv2D
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, SeparableConv2D, ReLU, Add, RNN, AbstractRNNCell
from tensorflow.keras.activations import tanh


class CNNCell(AbstractRNNCell):
    def __init__(self, units, args, **kwargs):
        self.units = units
        self.args = args
        self.block_size = args.block_size
        super(CNNCell, self).__init__(**kwargs)

    @property
    def state_size(self):
       return self.units

    def build(self, input_shape):
        self.model = CNNModel(self.args).get_model()
        self.built = True

    def call(self, inputs, states):
        # input: (batch, heightLR, widthLR, 3)

        # prev_output: (batch, heightHR, widthHR, 3)
        prev_output = states[0]

        # prev_output: (batch, heightLR, widthLR, 48)
        prev_output = tf.nn.space_to_depth(prev_output, self.block_size)

        # x: (batch, heightLR, widthLR, 51)
        x = tf.concat([prev_output, inputs], axis = -1)

        output = self.model(x)

        return output, output


class CNNModel:
    def __init__(self, args):
        # Image shape
        self.lr_height = args.lr_height
        self.lr_width = args.lr_width

        # Sequence
        self.sequence_length = args.sequence_length

        # Residual blocks
        self.num_res_blocks = args.num_res_blocks

        # Depthwise separable 2D convolution
        self.num_filters = args.num_filters
        self.kernel_size = args.kernel_size
        self.strides = args.strides
        self.padding = args.padding

        # Pixel Shuffle
        self.block_size = args.block_size

        # Upsampling factor (in each dimension)
        self.block_size = args.block_size


    def get_residual_block(self, input):
        x = Conv2D(filters = self.num_filters, kernel_size = (self.kernel_size, self.kernel_size), strides = (self.strides, self.strides), padding = self.padding)(input) # TODO Change to DWS
        x = ReLU()(x)
        x = Add()([input, x])
        return x

    def get_model(self):
        inputs = Input(shape = (self.lr_height, self.lr_width, 3 + 3 * 4 * 4))

        x = Conv2D(filters = self.num_filters, kernel_size = (self.kernel_size, self.kernel_size), strides = (self.strides, self.strides), padding = self.padding)(inputs)
        x = ReLU()(x)

        for _ in range(self.num_res_blocks - 1):
            x = self.get_residual_block(x)
        
        x = Conv2D(filters = 48, kernel_size = (self.kernel_size, self.kernel_size), strides = (self.strides, self.strides), padding = self.padding)(x)
        x = ReLU()(x)

        x = tf.nn.depth_to_space(input = x, block_size = self.block_size)

        outputs = (tanh(x) + 1) / 2

        model = Model(inputs = inputs, outputs = outputs)
        
        return model


class CustomRNN:
    def __init__(self, args):
        self.args = args
        self.sequence_length = args.sequence_length
        self.lr_height = args.lr_height
        self.lr_width = args.lr_width
        self.hr_height = args.hr_height
        self.hr_width = args.hr_width
        self.alpha = args.alpha

    def get_model(self):
        inputs = Input(shape = (self.sequence_length, self.lr_height, self.lr_width, 3))
        rnn = RNN(CNNCell(units = tf.TensorShape((self.hr_height, self.hr_width, 3)), args = self.args), return_sequences = True)(inputs)
        
        model = Model(inputs = inputs, outputs = rnn)

        return model