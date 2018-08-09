import numpy as np
from dataset._base_ds import BaseDS
import math

class DummyDS(BaseDS):
    def __init__(self, nclass,
                 image_size, max_label_len, train_count=12345, test_count=890):
        super(DummyDS, self).__init__(nclass, image_size)
        self.max_label_len = max_label_len
        self.train_count = train_count
        self.test_count = test_count

    def train_samples(self):
        sample_count = self.train_count
        samples = self._generate_sample_collection_(sample_count)
        return samples

    def test_samples(self):
        sample_count = self.test_count
        samples = self._generate_sample_collection_(sample_count)
        return samples

    def get_item(self, context, samples, key):
        input = np.random.uniform(-1, 1, self.image_size)
        label = samples[key]

        # input_length = self.image_size[1] // self.strides
        # label_length = len(label)
        return [input, label, key]

    def get_all_keys(self, samples):
        return samples.keys()

    def _generate_sample_collection_(self, sample_count):
        max_label_len = self.max_label_len
        dic = {}
        for i in range(sample_count):
            name = 'sample.' + str(i)
            label_len = i % (max_label_len) + 1
            label = np.random.randint(0,self.nclass-1,(label_len),dtype=np.int32)
            dic[name] = label
        return dic

if __name__ == '__main__':
    img_size = (32, 280, 1)

    ds = DummyDS(nclass=2500, max_label_len=10, image_size=img_size)
    gen, gen_sz = ds.train_gen()
    pass
    item_count = 0
    for item in gen:
        item_count += 1
        pass
    pass

    batch_size = 128
    batch_gen, bat_gen_sz = ds.train_batch_gen(batch_size)
    b_item_count = 0
    keys={}
    for batch_item in batch_gen:
        # item_count
        batch_keys = batch_item.keys
        b_item_count += len(batch_keys)
        for key in batch_keys:
            keys[key] = key

    assert (math.ceil(item_count / batch_size) * batch_size) == bat_gen_sz
    assert item_count == len(keys)
    pass

