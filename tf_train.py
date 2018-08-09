import logging
import os
import time
import tensorflow as tf

from config import Config
from dataset.dummy_ds import DummyDS
from helpers import delta_time_str

logger = logging.getLogger('Traing for OCR using CRNN')
logger.setLevel(logging.INFO)

tf.app.flags.DEFINE_boolean('restore', False, 'whether to restore from the latest checkpoint')
tf.app.flags.DEFINE_string('checkpoint_dir', './runtime/checkpoint/', 'the checkpoint dir')
tf.app.flags.DEFINE_float('initial_learning_rate', 5e-4, 'inital lr')

tf.app.flags.DEFINE_float('output_keep_prob', 0.8, 'output_keep_prob in lstm')
tf.app.flags.DEFINE_integer('num_epochs', 10000, 'maximum epochs')
# tf.app.flags.DEFINE_integer('batch_size', 40, 'the batch_size')
tf.app.flags.DEFINE_integer('save_steps', 1000, 'the step to save checkpoint')
tf.app.flags.DEFINE_float('leakiness', 0.01, 'leakiness of lrelu')
tf.app.flags.DEFINE_integer('validation_steps', 2000, 'the step to validation')

tf.app.flags.DEFINE_float('decay_rate', 0.98, 'the lr decay rate')
tf.app.flags.DEFINE_float('beta1', 0.9, 'parameter of adam optimizer beta1')
tf.app.flags.DEFINE_float('beta2', 0.999, 'adam parameter beta2')

tf.app.flags.DEFINE_integer('decay_steps', 10000, 'the lr decay_step for optimizer')
tf.app.flags.DEFINE_float('momentum', 0.9, 'the momentum')

tf.app.flags.DEFINE_string('train_dir', './imgs/train/', 'the train data dir')
tf.app.flags.DEFINE_string('val_dir', './imgs/val/', 'the val data dir')
tf.app.flags.DEFINE_string('infer_dir', './imgs/infer/', 'the infer data dir')
tf.app.flags.DEFINE_string('log_dir', './runtime/log', 'the logging dir')
# tf.app.flags.DEFINE_string('mode', 'train', 'train, val or infer')
tf.app.flags.DEFINE_integer('num_gpus', 0, 'num of gpus')

FLAGS = tf.app.flags.FLAGS

class OCR(object):
    def __init__(self, feature_shape, nclass, training):
        self.feature_shape = feature_shape
        self.nclass = nclass
        self.training = training
        self.input_shape = feature_shape

    def build_graph(self):
        self._build_model()
        self._build_train_op()
        self.merged_summay = tf.summary.merge_all()

    def _build_model(self):
        from model_factory import ModelFactory
        model = ModelFactory().inference(input_shape=self.input_shape,
                                              nclass=self.nclass)
        self.inputs, self.logits = model.input, model.output
        self.labels = tf.sparse_placeholder(tf.int32)
        logits_shape = tf.shape(self.logits)
        self.seq_len = tf.fill([logits_shape[0],], logits_shape[1])
        pass

    def _build_train_op(self):
        self.global_step = tf.train.get_or_create_global_step()
        self.loss = tf.nn.ctc_loss(labels=self.labels,
                                   inputs=self.logits,
                                   sequence_length=self.seq_len,
                                   time_major=False)
        self.cost = tf.reduce_mean(self.loss)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=Config.LEARNING_RATE)
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

class _GenConvertor:
    def __init__(self):
        pass
    def convert(self, plain):
        features = plain.features
        from helpers import sparse_tuple_from_label
        sparse_labels = sparse_tuple_from_label(plain.labels)
        return (features, sparse_labels, plain.labels)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)

    img_size = (32, 280, 1)
    feature_shape = (img_size[0], None, img_size[2])
    batch_size = 4
    nclass = 3000
    ds = DummyDS(nclass=nclass, max_label_len=10, image_size=img_size, train_count=1000, test_count=100)

    gen_conv = _GenConvertor()
    train_gen, train_len = ds.train_batch_gen(batch_size, post_op=gen_conv.convert, loops=None)
    test_gen, test_len = ds.test_batch_gen(batch_size, post_op=gen_conv.convert, loops=None)

    num_batches_per_epoch = int(train_len / batch_size)  # example: 100000/100
    num_batches_per_epoch_val = int(test_len / batch_size)  # example: 10000/100

    model = OCR(feature_shape=feature_shape, nclass=nclass, training=True)
    model.build_graph()

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        trainable_variables = sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        print('=============================begin training=============================')
        for cur_epoch in range(FLAGS.num_epochs):
            train_cost = 0
            epoch_time_start = time.time()

            # the training part
            for cur_batch in range(num_batches_per_epoch):
                batch_start_time = time.time()

                batch_item = next(train_gen)
                batch_inputs = batch_item[0]
                batch_labels = batch_item[1]

                feed = {model.inputs: batch_inputs,
                        model.labels: batch_labels}

                logits, batch_cost, lbs, seq_len, step, _ = \
                    sess.run([model.logits, model.cost, model.labels, model.seq_len, model.global_step, model.train_op], feed)

                train_cost += batch_cost * batch_size

                batch_cost_time = time.time() - batch_start_time
                if (cur_batch) % 2 == 0 or cur_batch == (num_batches_per_epoch - 1):
                    epoch_time_cost = time.time() - epoch_time_start
                    progress = (cur_batch + 1) / num_batches_per_epoch
                    epoch_time_total = epoch_time_cost / progress

                    print("Epoch.{}".format(cur_epoch),
                          'batch.{0}({1:.2%})'.format(cur_batch, progress),
                          ': time:{:2.2f}s'.format(time.time() - batch_start_time),
                          ', loss:{:2.2f} '.format(batch_cost),
                          '[Used: {0}, Total: {1}, Remaining: {2}]'.format(delta_time_str(epoch_time_cost),
                                                                           delta_time_str(epoch_time_total),
                                                                           delta_time_str(epoch_time_total-epoch_time_cost)))
