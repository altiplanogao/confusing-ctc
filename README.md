# confusing-ctc

I read code (https://github.com/YCG09/chinese_ocr), and made some change (replace densenet by classic cnn), it works.

And now, I'm tring to update it from keras to tensorflow. But the loss doesn't decrease in my tensorflow version.

This repository contains both versions (Keras and Tensorflow).

Problem is not solved currently, and I'll make update if any progress.


## Code explain
dataset: contains 2 kind of dataset (provide data in generator way):
* fs_ds.py: dataset base on image files and labels file (images in shape: 32*280, each image contains several characters)
* dummy_ds.py: mock images in shape 32*280 by calling np.random(). (fake data which makes nonsense. But it requires no disk space, and enough to represent my problem)


model_factory.py: Model Factory

### Trainers:
* keras_train.py: train using keras (works expected)
* tf_train.py: train using tensorflow (works unexpected)

They:
 * share same model factory
 * both use 'adam' as optimizer, while I also noticed that keras and tf have different adam implementation. 

### Sample output:
Keras version: (the final loss would be arround 50)
```
  1/250 [..............................] - ETA: 11:24 - loss: 248.5603 - acc: 0.0000e+00
  2/250 [..............................] - ETA: 8:56 - loss: 140.4536 - acc: 0.0000e+00 
  3/250 [..............................] - ETA: 8:02 - loss: 131.6902 - acc: 0.0000e+00
  4/250 [..............................] - ETA: 7:34 - loss: 116.7303 - acc: 0.0000e+00

.....

 17/250 [=>............................] - ETA: 6:25 - loss: 83.4287 - acc: 0.0000e+00
 ```
 
 Tensorflow version: (loss value keeps moving arround 260, even after a very long time)
``` 
Epoch.0 batch.0(0.40%) : time:1.67s , loss:259.88  , learn_rate:0.00050 [Used: 00:00:01.7, Total: 00:06:58.0, Remaining: 00:06:56.3]
Epoch.0 batch.2(1.20%) : time:1.13s , loss:257.59  , learn_rate:0.00050 [Used: 00:00:04.0, Total: 00:05:33.3, Remaining: 00:05:29.3]
Epoch.0 batch.4(2.00%) : time:1.14s , loss:260.16  , learn_rate:0.00050 [Used: 00:00:06.3, Total: 00:05:13.3, Remaining: 00:05:07.0]
Epoch.0 batch.6(2.80%) : time:1.13s , loss:261.42  , learn_rate:0.00050 [Used: 00:00:08.5, Total: 00:05:03.6, Remaining: 00:04:55.1]
......
Epoch.0 batch.24(10.00%) : time:1.33s , loss:260.79  , learn_rate:0.00050 [Used: 00:00:30.1, Total: 00:05:01.3, Remaining: 00:04:31.1]

```
