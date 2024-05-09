# image-classification-improvement-task

## Data set Analysis
Looking through the dataset, I realized the dataset is high imbalance. Since the task was to use accuracy as a metric. 
The first thing was to perform a class balance step. This was simply done by augmenting the low count classes. 
I could have reduced the higher classes in count, but that affects the downstream training step for different models as
We tend to have lower sample sets. 

Class  differences:
1          0
2          6
0          6
3          46
Class  percentages:
1          0.296774
2          0.277419
0          0.277419
3          0.148387

After the initial augmentation step, we are at a balanced dataset
Class  differences:
3         0
0         0
2         0
1         0
Class  percentages:
3         0.25
0         0.25
2         0.25
1         0.25

## Data Augmentation
410 images are in total available, this set is deemed very low , even splitting this data into train/test will 
produce a network with no meaningful results.Hence a second step of argumentation is performed, 
which includes, flips, erode, dilate etc. to produce the second dataset that contains 2296 images in total. 
This is the final dataset used to perform training.

## Training 
I have trained two different architectures, a simple MLP and a simple RNN network to produce results.
For both cases I have used raw image features and GIST features. The results are very interesting. 
which are further shown below
During training, the training dataset is split into 70%,30% ratio for training and testing. I have made sure
that the split is stratified so that each class has a near equal chance of being used in training and testing.

## Results

# MLP Model

- Raw MLP Train accuracy: **44.59546661376953%**
- Test Loss: **0.824209**

## Test Accuracy by Class

- Test Accuracy of 0: **85% (138/162)**
- Test Accuracy of 1: **75% (133/176)**
- Test Accuracy of 2: **5% (9/167)**
- Test Accuracy of 3: **0% (0/158)**

## Overall Test Accuracy

- RAW MLP Test Accuracy (Overall): **42% (280/663)**

**GIST MLP Train accuracy: 69.57928466796875%
Test Loss: 0.374313
Test Accuracy of     0: 71% (116/162)
Test Accuracy of     1: 44% (79/176)
Test Accuracy of     2: 71% (119/167)
Test Accuracy of     3: 86% (136/158)
GIST MLP Test Accuracy (Overall): 67% (450/663)
---------------------------------------------------------------------**
RNN Model
RAW RNN Train accuracy: 91.84465789794922%
Test Loss: 0.248792
Test Accuracy of     0: 90% (146/162)
Test Accuracy of     1: 62% (110/176)
Test Accuracy of     2: 70% (117/167)
Test Accuracy of     3: 92% (146/158)
RAW RNN Test Accuracy (Overall): 78% (519/663)

GIST RNN Train accuracy: 89.19093322753906%
Test Loss: 0.096881
Test Accuracy of     0: 92% (150/162)
Test Accuracy of     1: 87% (154/176)
Test Accuracy of     2: 79% (132/167)
Test Accuracy of     3: 95% (151/158)
GIST RNN Test Accuracy (Overall): 88% (587/663)

# Conclusion
Using two different models with two different features, it is evident that a model with classical features such as GISt can outdo a simple model with raw features. In both cases for RNN and MLP, if we keep the raw features model
as baseline, we can see from the results that GIST features not only reduce overfitting, but also increase the accuracy in the test set - a clear indication is the per class accuracy. Hence we can safely conclude that using a simpler network with a change of the underlying features, 
we can produce a network that is better. 

A jupyter notebook (Evaluate_TestSet.ipynb) is also available, the code performs an evaluation on the "test" folder. The outputs are csv files labeling
the input for all 4 neural networks. 
