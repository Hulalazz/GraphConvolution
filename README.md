# GraphConvolution

## Info

PyTorch implementation for [**Kipf & Welling (2017). Semi-Supervised Classification with Graph Convolutional Networks. ICLR**](http://arxiv.org/abs/1609.02907). The data comes from the first author's repo [here](https://github.com/tkipf/gcn/tree/master/gcn/data).

## Dependencies

networkx, numpy, pickle, torch (0.3.1), scipy.

## Usage

To train the model, run `run_train.sh` with correctly chosen parameters as per your need.
Example output on the Cora dataset:
```
Iteration 50:

Train loss = 1.9893128 (at step 50)
Train/Valid/Test accuracy: 0.2929 | 0.1800 | 0.1790

Iteration 100:

Train loss = 1.8589772 (at step 100)
Train/Valid/Test accuracy: 0.8143 | 0.5220 | 0.5170

Iteration 150:

Train loss = 1.7107995 (at step 150)
Train/Valid/Test accuracy: 0.9429 | 0.6540 | 0.6930

Iteration 200:

Train loss = 1.5520712 (at step 200)
Train/Valid/Test accuracy: 0.9714 | 0.7040 | 0.7420

Iteration 250:

Train loss = 1.3880322 (at step 250)
Train/Valid/Test accuracy: 0.9714 | 0.7200 | 0.7550

Iteration 300:

Train loss = 1.2250416 (at step 300)
Train/Valid/Test accuracy: 0.9786 | 0.7480 | 0.7670

Iteration 350:

Train loss = 1.0695306 (at step 350)
Train/Valid/Test accuracy: 0.9857 | 0.7520 | 0.7780

Iteration 400:

Train loss = 0.9264522 (at step 400)
Train/Valid/Test accuracy: 0.9857 | 0.7660 | 0.7850

Iteration 450:

Train loss = 0.79867464 (at step 450)
Train/Valid/Test accuracy: 0.9857 | 0.7660 | 0.7850

Iteration 500:

Train loss = 0.68706673 (at step 500)
Train/Valid/Test accuracy: 0.9857 | 0.7700 | 0.7920

Iteration 550:

Train loss = 0.5910516 (at step 550)
Train/Valid/Test accuracy: 0.9857 | 0.7760 | 0.7980

Iteration 600:

Train loss = 0.5092095 (at step 600)
Train/Valid/Test accuracy: 0.9857 | 0.7760 | 0.7990

Iteration 650:

Train loss = 0.43980482 (at step 650)
Train/Valid/Test accuracy: 1.0000 | 0.7760 | 0.7990

Training accuracy reaches 1.0, training done!
```

## Comments

Accuracy within 2% of that is reported in the original work. The model is sensitive to hyperparameter tuning. 
