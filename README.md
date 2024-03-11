# Neural Networks and Learning Machines

## Double-Moon Structures
```matlab
radius = 10; width = 4; theta = 25; distance = -4;
N = 400; train2total = 0.75; drawPatterns = true;

[train, test] = doubleMoonStructure( radius, width, theta, distance, ...
                                     N, train2total, drawPatterns );
```
![Double-Moon Structure Train Dataset](./Documentations/Figures/train-dataset.jpg "Double-Moon Structure Train Dataset")
![Double-Moon Structure Test Dataset](./Documentations/Figures/test-dataset.jpg "Double-Moon Structure Test Dataset")

## Rosenblatt's Perceptron
### Train Perceptron
```matlab
eta = 1;            % Learning-Rate Parameter
w0 = [ 0, 0, 0 ];   % Initialize the Weights

% Training the Perceptron
weights = trainPerceptron(train, eta, w0, true);
```
![Train Perceptron](./Documentations/Figures/train-perceptron-gif.gif "Train Perceptron")
### Test Perceptron
```matlab
% Validation on the Test Dataset
% Concluding the Accuracy of the Decision Boundry
[class1, class2, accuracy] = testPerceptron(test, weights);
```
![Tesst Perceptron](./Documentations/Figures/test-perceptron.jpg "Test Perceptron")