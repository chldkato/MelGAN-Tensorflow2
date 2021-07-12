import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import LeakyReLU, Conv1DTranspose, AveragePooling1D, Activation, Conv1D
from models.modules import residual_stack
from util.hparams import *


class Generator(Model):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.layer = Sequential([
            Conv1D(512, kernel_size=7, padding='same')
        ])
        
        factor = [8, 8, 2, 2]
        filters = 256
        
        for f in factor:
            self.layer.add(LeakyReLU(0.2))
            self.layer.add(
                    Conv1DTranspose(filters, kernel_size=2 * f, strides=f, padding='same')
            )
            filters //= 2
            for d in range(3):
                self.layer.add(residual_stack(filters, 3 ** d))
                
        self.layer.add(LeakyReLU(0.2))
        self.layer.add(Conv1D(1, kernel_size=7, padding='same'))
        self.layer.add(Activation('tanh'))
        
        
    def call(self, mel_input):
        return self.layer(mel_input)
    
    
class Discriminator(Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layer = [
            Sequential([
                Conv1D(16, kernel_size=15, padding='same'),
                LeakyReLU(0.2)
            ]),
            Sequential([
                Conv1D(64, kernel_size=41, strides=4, padding='same', groups=4),
                LeakyReLU(0.2)
            ]),
            Sequential([
                Conv1D(256, kernel_size=41, strides=4, padding='same', groups=16),
                LeakyReLU(0.2)
            ]),
            Sequential([
                Conv1D(1024, kernel_size=41, strides=4, padding='same', groups=64),
                LeakyReLU(0.2)
            ]),
            Sequential([
                Conv1D(1024, kernel_size=41, strides=4, padding='same', groups=256),
                LeakyReLU(0.2)
            ]),
            Sequential([
                Conv1D(1024, kernel_size=5, padding='same'),
                LeakyReLU(0.2)
            ]),
            Sequential([
                Conv1D(1, kernel_size=3, padding='same'),
            ])
        ]    
    
    def call(self, x):
        result = []
        for layer in self.layer:
            x = layer(x)
            result.append(x)
        return result    
        

class MultiScaleDiscriminator(Model):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        
        self.layer = [
            Discriminator() for _ in range(3)     
        ]
        
        self.avgpool = AveragePooling1D(pool_size=4, strides=2, padding='same')
    
   
    def call(self, x):
        result = []
        for i, layer in enumerate(self.layer):
            result.append(layer(x))
            if i <= 1:
                x = self.avgpool(x)        
        return result
    