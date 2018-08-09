import math
import numpy as np
import random

class batch_item:
    def __init__(self, features, labels, keys):
        self.features = features
        self.labels = labels
        self.keys = keys

class BaseDS:
    def __init__(self, nclass,
                 image_size):
        self.nclass = nclass
        self.image_size = image_size

    def train_samples(self):
        raise NotImplementedError
        # samples = [1,2,3,4,5,6,7,8,9]
        # return samples

    def test_samples(self):
        raise NotImplementedError
        # samples = [9, 10]
        # return samples

    def get_item(self, context, samples, sample_key):
        raise NotImplementedError
        # feature = np.random.uniform(-1, 1, self.image_size)
        # label = samples[key]
        # return [features, labels, key]

    def get_all_keys(self, samples):
        raise NotImplementedError

    def train_gen(self, post_op=None):
        samples = self.train_samples()
        return self._generator_(samples, post_op=post_op), len(samples)

    def test_gen(self, post_op=None):
        samples = self.test_samples()
        return self._generator_(samples, post_op=post_op), len(samples)

    def train_batch_gen(self, batch_size, post_op=None, loops=1):
        samples = self.train_samples()
        shuffled_keys = self._shuffle_for_batch_(self.get_all_keys(samples), batch_size)
        return self._batch_generator_(samples, shuffled_keys, batch_size=batch_size, post_op=post_op, loops=loops), len(shuffled_keys)

    def test_batch_gen(self, batch_size, post_op=None, loops=1):
        samples = self.test_samples()
        shuffled_keys = self._shuffle_for_batch_(self.get_all_keys(samples), batch_size)
        return self._batch_generator_(samples, shuffled_keys, batch_size=batch_size, post_op=post_op, loops=loops), len(shuffled_keys)

    def _open_gen_context_(self) -> object:
        return None

    def _close_gen_context(self, context):
        pass

    def _generator_(self, samples, post_op):
        ctx = self._open_gen_context_()
        for sample in samples:
            item = self.get_item(ctx, samples, sample)
            if post_op is None:
                yield item
            else:
                yield post_op(item)
        self._close_gen_context(ctx)

    def _shuffle_for_batch_(self, keys, batch_size):
        keys = list(keys)
        key_count = len(keys)
        batch_remain = key_count % batch_size
        batch_require = 0
        if batch_remain != 0:
            batch_require = batch_size - batch_remain
        result = keys.copy()
        if batch_require != 0:
            key_copy = keys.copy()
            random.shuffle(key_copy)
            result.extend(key_copy[0:batch_require])
        random.shuffle(result)
        return result

    def _batch_generator_(self, samples, shuffled_keys, batch_size, post_op, loops):
        imagesize = self.image_size
        b_features = np.zeros((batch_size, imagesize[0], imagesize[1], imagesize[2]), dtype=np.float)
        b_labels = [None] * batch_size
        b_keys = [None] * batch_size

        key_count = len(shuffled_keys)
        batches = key_count // batch_size
        ctx = self._open_gen_context_()
        loop_index = 0
        while ((loops is None) or (loops < 0)) or loop_index < loops:
            loop_index += 1
            for b in range(batches):
                start_key_index = b * batch_size
                for i in range(batch_size):
                    key = shuffled_keys[i + start_key_index]
                    item = self.get_item(ctx, samples, key)

                    b_features[i] = item[0]
                    b_labels[i] = item[1]
                    b_keys[i] = key

                inputs = batch_item(features=b_features,
                                    labels=b_labels,
                                    keys=b_keys)
                if post_op is None:
                    yield inputs
                else:
                    yield post_op(inputs)
        self._close_gen_context(ctx)
