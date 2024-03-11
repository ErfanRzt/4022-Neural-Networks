# Neural Networks and Learning Machines

## Double-Moon Structures
```matlab
radius = 10; width = 4; theta = 25; distance = -4;
N = 400; train2total = 0.75; drawPatterns = true;

[train, test] = doubleMoonStructure( radius, width, theta, distance, ...
                                     N, train2total, drawPatterns );
```
<div style="text-align: center;">
  <img src="./Documentations/Figures/train-dataset.jpg" alt="TrainDataset" width="500"/>
  <img src="./Documentations/Figures/test-dataset.jpg" alt="TestDataset" width="500"/>
</div>

## Rosenblatt's Perceptron
### Train Perceptron
```matlab
eta = 1;            
w0 = [ 0, 0, 0 ];   

weights = trainPerceptron(train, eta, w0, true);
```
<div style="text-align: center;">
  <img src="./Documentations/Figures/train-perceptron-gif.gif" alt="TrainPerceptron" width="500"/>
</div>

### Test Perceptron
```matlab
[class1, class2, accuracy] = testPerceptron(test, weights);
```
<div style="text-align: center;">
  <img src="./Documentations/Figures/test-perceptron.jpg" alt="TestPerceptron" width="500"/>
</div>
