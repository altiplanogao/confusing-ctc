import os
import os.path as path

import tensorflow
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, LearningRateScheduler
from tensorflow.python.keras.layers import Lambda

from tensorflow.python.keras import backend as K, Input, Model

import numpy as np

from dataset.dummy_ds import DummyDS
from model_factory import ModelFactory

# Convert the generator in Keras acceptable
class _GenConvertor:
    def __init__(self):
        self.invalid_class = -1
    def convert(self, plain):
        from config import Config
        import functools
        label_w = functools.reduce((lambda a, b: a if a > b else b), map(lambda label: len(label), plain.labels))
        label_h = len(plain.labels)
        _labels = np.full(shape=[label_h, label_w], dtype=np.int32, fill_value=self.invalid_class)
        _label_lens = np.zeros((batch_size, 1), dtype=np.int32)

        for i in range(label_h):
            item_label = plain.labels[i]
            item_label_len = len(item_label)
            _labels[i, :item_label_len] = item_label
            _label_lens[i] = item_label_len

        inputs = {Config.INPUTS: plain.features[0:label_h],
                  'the_labels': _labels,
                  'label_length': _label_lens
                  }
        outputs = {'ctc': np.zeros([label_h])}
        return (inputs, outputs)

class ModelTrainer:
    def __init__(self, h_size, nclass, model_factory,
                 train_gen, train_steps_per_epoch,
                 validation_gen, validation_steps,
                 batch_size):
        self.h_size = h_size
        self.nclass = nclass
        self.model_factory = model_factory
        self.train_gen = train_gen
        self.train_steps_per_epoch = train_steps_per_epoch
        self.validation_gen = validation_gen
        self.validation_steps = validation_steps
        self.batch_size = batch_size
        self._build_model_()

    def _build_model_(self):
        input_shape=(self.h_size, None, 1)
        basemodel = self.model_factory.inference(input_shape=input_shape, nclass=self.nclass)
        basemodel.summary()
        self.basemodel = basemodel

        labels = Input(name='the_labels', shape=[None], dtype='int32')
        label_length = Input(name='label_length', shape=[1], dtype='int32')

        # labels = Input(name=Config.LABELS, shape=(None,), sparse=True, dtype='int32')

        base_input = basemodel.input
        base_output = basemodel.output

        def ctc_lambda_func(args):
            import tensorflow as tf
            base_output, labels, label_length = args
            base_output_shape = tf.shape(base_output)
            sequence_length = tf.fill([base_output_shape[0],], base_output_shape[1])
            return K.ctc_batch_cost(labels, base_output, sequence_length, label_length)
        loss = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([base_output, labels, label_length])

        train_model = Model(inputs=[base_input, labels, label_length], outputs=loss)
        def loss_func(y_true, y_pred):
            return y_pred
        train_model.compile(loss={'ctc': loss_func}, optimizer='adam', metrics=['accuracy'])
        self.train_model = train_model

    def train(self):
        epoches = 10
        lr_schedule = lambda epoch: 0.0005 * 0.4 ** epoch
        learning_rate = np.array([lr_schedule(i) for i in range(epoches)])
        changelr = LearningRateScheduler(lambda epoch: float(learning_rate[epoch]))
        earlystop = EarlyStopping(monitor='val_loss', patience=2, verbose=1)

        print('-----------Start training-----------')
        self.train_model.fit_generator(generator=self.train_gen,
                            steps_per_epoch=self.train_steps_per_epoch // self.batch_size,
                            epochs=epoches,
                            initial_epoch=0,
                            validation_data=self.validation_gen,
                            validation_steps=self.validation_steps // self.batch_size,
                            callbacks=[earlystop, changelr])
        pass


if __name__ == '__main__':
    img_size = (32, 280, 1)
    max_label_len = 10
    batch_size = 4
    nclass = 3000

    model_factory = ModelFactory()

    convertor = _GenConvertor()
    ds = DummyDS(nclass=nclass, max_label_len=max_label_len, image_size=img_size, train_count=1000, test_count=100)
    train_gen, train_len = ds.train_batch_gen(batch_size,convertor.convert, loops=None)
    test_gen, test_len = ds.test_batch_gen(batch_size,convertor.convert, loops=None)

    trainer = ModelTrainer(img_size[0], nclass=nclass, model_factory=model_factory,
                           train_gen=train_gen, train_steps_per_epoch=train_len,
                           validation_gen=test_gen, validation_steps=test_len,
                           batch_size=batch_size)
    trainer.train()

    pass
