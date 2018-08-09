# confusing-ctc

I read code (https://github.com/YCG09/chinese_ocr), and made some change (replace densenet by classic cnn), it works.

And now, I'm trying to update it from keras to tensorflow. But the loss doesn't decrease in my tensorflow version.

This repository contains both versions (Keras and Tensorflow).

Problem is not solved currently, and I'll make update if any progress.

If you have any idea, please reply to my question [here](https://stackoverflow.com/questions/51766943/ctc-loss-doesnt-decrease-using-tensorflow-while-it-decreases-using-keras). Many may thanks.

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
 
 Tensorflow version: (loss value keeps moving around 230, even after a very long time)
``` 
Epoch.0 batch.0(0.40%) : time:1.64s , loss:256.49  [Used: 00:00:01.6, Total: 00:06:50.2, Remaining: 00:06:48.5]
Epoch.0 batch.2(1.20%) : time:1.22s , loss:238.10  [Used: 00:00:04.1, Total: 00:05:38.6, Remaining: 00:05:34.5]
Epoch.0 batch.4(2.00%) : time:1.30s , loss:235.42  [Used: 00:00:06.7, Total: 00:05:36.7, Remaining: 00:05:30.0]
Epoch.0 batch.6(2.80%) : time:1.33s , loss:237.00  [Used: 00:00:09.4, Total: 00:05:36.5, Remaining: 00:05:27.1]
Epoch.0 batch.8(3.60%) : time:1.33s , loss:236.71  [Used: 00:00:12.2, Total: 00:05:38.6, Remaining: 00:05:26.4]
......
Epoch.0 batch.42(17.20%) : time:1.52s , loss:233.79  [Used: 00:00:55.1, Total: 00:05:20.3, Remaining: 00:04:25.2]

```
