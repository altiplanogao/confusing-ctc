from tensorflow import keras

class ConvDef:
    def __init__(self, filters, ksize=(3,3), strides=(1,1), padding='same'):
        self.ksize = (ksize[0], ksize[1])
        self.strides = (strides[0], strides[1])
        self.padding = padding
        self.filters = filters

class ModelFactory:
    def __init__(self, dropout_rate = 0.3):
        self.dropout_rate = dropout_rate
        pass

    def _convRelu(self, model, conv_args, batch_norm=True, drop_out=False):
        _weight_decay = 1e-4
        if batch_norm:
            bn = keras.layers.BatchNormalization(axis=-1, epsilon=1.1e-5)
            model.add(bn)
        conv = keras.layers.Conv2D(conv_args.filters, conv_args.ksize, strides=conv_args.strides,
                                   kernel_initializer='he_normal', padding=conv_args.padding,
                                   kernel_regularizer=keras.regularizers.l2(_weight_decay),
                                   use_bias=False)
        model.add(conv)
        act = keras.layers.Activation('relu')
        model.add(act)

        if drop_out:
            model.add(keras.layers.Dropout(self.dropout_rate))

    def bidirectionalLSTM(self, model, out_dim):
        lstm_input = model.output
        model.add(keras.layers.Bidirectional(keras.layers.LSTM(out_dim, return_sequences=True)))
        # model.add(Bidirectional(LSTM(hidden, return_sequences=True)))

    def inference(self, input_shape, nclass):
        model = keras.Sequential()
        from config import Config
        input_layer = keras.layers.InputLayer(input_shape=input_shape, name=Config.INPUTS)
        model.add(input_layer)
        # conv1, conv2
        self._convRelu(model, ConvDef(filters=64, ksize=(3, 3), strides=(1, 1), padding='same'))
        self._convRelu(model, ConvDef(filters=64, ksize=(3, 3), strides=(1, 1), padding='same'), drop_out=True)
        model.add(keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        # conv3, conv4
        self._convRelu(model, ConvDef(filters=128, ksize=(3, 3), strides=(1, 1), padding='same'), True)
        self._convRelu(model, ConvDef(filters=128, ksize=(3, 3), strides=(1, 1), padding='same'), drop_out=True)
        model.add(keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))
        # conv5, conv6
        self._convRelu(model, ConvDef(filters=256, ksize=(3, 3), strides=(1, 1), padding='same'), True)
        self._convRelu(model, ConvDef(filters=256, ksize=(3, 3), strides=(1, 1), padding='same'), drop_out=True)
        model.add(keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))
        # conv7, conv8
        self._convRelu(model, ConvDef(filters=512, ksize=(4, 4), strides=(4, 1), padding='same'), False)
        self._convRelu(model, ConvDef(filters=512, ksize=(1, 1), strides=(1, 1), padding='valid'), False)

        model.add(keras.layers.Permute((2, 1, 3), name='permute'))
        model.add(keras.layers.TimeDistributed(keras.layers.Flatten(), name='flatten'))

        # self.bidirectionalLSTM(384)
        # self.bidirectionalLSTM(256)

        model.add(keras.layers.Dense(nclass, name=Config.OUTPUTS, activation='softmax'))
        return model

if __name__ == '__main__':
    input_shape = (32,100, 1)
    modelFactory = ModelFactory()
    modelFactory.inference(input_shape, 3501)
    pass
