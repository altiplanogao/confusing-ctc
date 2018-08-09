import os
import numpy as np
from PIL import Image
from dataset import BaseDS

class FileSystemDS(BaseDS):
    def __init__(self, base_dir, images, train_labels_file, test_labels_file, nclass,
                 image_size=(32,280,1)):
        super(FileSystemDS, self).__init__(nclass, image_size)
        self.images = os.path.join(base_dir, images)
        self.train_labels = os.path.join(base_dir, train_labels_file)
        self.test_labels = os.path.join(base_dir, test_labels_file)

    def train_samples(self):
        samples = self._read_samples_(self.train_labels)
        return samples

    def test_samples(self):
        samples = self._read_samples_(self.test_labels)
        return samples

    def get_item(self, context, samples, key):
        img1 = Image.open(os.path.join(self.images, key)).convert('L')
        img = np.array(img1, 'f') / 128.0 - 1.0

        feature = np.expand_dims(img, axis=2)
        label = samples[key]
        return [feature, label, key]

    def get_all_keys(self, samples):
        image_files = [i for i, j in samples.items()]
        return image_files

    def _read_samples_(self, labels_file):
        res = []
        with open(labels_file, 'r') as f:
            lines = f.readlines()
            for i in lines:
                res.append(i.strip())
        dic = {}
        for i in res:
            p = i.split(' ')
            label_cats = [int(i) - 1 for i in p[1:]]
            dic[p[0]] = label_cats
        return dic
