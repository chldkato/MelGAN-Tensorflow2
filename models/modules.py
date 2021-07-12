import tensorflow as tf
from tensorflow_addons.layers import WeightNormalization
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Conv1D, LeakyReLU


class residual_stack(Model):
    def __init__(self, filters, d):
        super(residual_stack, self).__init__()
        self.layer = Sequential([
            LeakyReLU(0.2),
            Conv1D(filters, kernel_size=3, padding='same', dilation_rate=d),
            LeakyReLU(0.2),
            Conv1D(filters, kernel_size=1)
        ])
        
        self.conv = Conv1D(filters, kernel_size=1)
        
    def call(self, input_data):
        x = self.layer(input_data)
        return x + self.conv(input_data)
