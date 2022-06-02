
from keras.layers import (Activation, BatchNormalization, Dense, Flatten,
                          Input, LeakyReLU)
from keras.layers.convolutional import Conv3D
from keras.models import Model
from keras.optimizers import Adam


class RootClassificationModel(object):
    def __convolution_block(self, tensor, **additional_params):
        base_params = {'filters': 16,
                       'kernel_size' : 2,
                       'strides': 2,
                       'padding': 'same',
                       'kernel_initializer': 'he_normal'}
        
        base_params.update(additional_params)
        tensor = Conv3D(** base_params)(tensor)
        tensor = BatchNormalization()(tensor)
        tensor = LeakyReLU(0.2)(tensor)

        return tensor

    def __init__(self, pretrained_weights = None, input_shape = (17,17,17,1), filter_count=16):
        inputs = Input(input_shape)

        conv1 = self.__convolution_block(inputs, filters=filter_count, padding='valid')
        conv2 = self.__convolution_block(conv1, filters=filter_count*2, padding='same')
        conv3 = self.__convolution_block(conv2, filters=filter_count*4, padding='same')
        conv4 = self.__convolution_block(conv3, filters=filter_count*8, padding='same')
        conv5 =  Flatten()(conv4)
        output = Activation('sigmoid')(Dense(1)(conv5))

        self.model = Model(inputs = [inputs], outputs = [output])
        self.model.compile(optimizer = Adam(lr = 1e-5), loss='mean_squared_error', metrics=["accuracy"])

        if pretrained_weights:
            self.model.load_weights(pretrained_weights)

    def get(self):
        return self.model
        
    def summary(self):
        self.model.summary()

if __name__ == '__main__':
    model = RootClassificationModel()
    model.summary()
