function [weights] = trainPerceptron(train, eta, w0)
trainClassSize = size(train);
trainPerceptron = [ train(:,1:2)    ones(trainClassSize(1),1)    ;   % train dataset with desired values;
                    train(:,3:4)    -1*ones(trainClassSize(1),1) ];  % desired values for C1 = 1, and for C2 = -1

weights = w0';  % Initialize the Weights

% Activate the perceptron by applying input
% vector x(n) and desired response d(n)
for i = 1:trainClassSize(1)
    % Alternating Sampling: 
    % Applying inputs from different category at each iteration
    for j = 1:2         
        index = i + ((j - 1) * trainClassSize(1));
        x = [1, trainPerceptron(index, 1:2)]';
        d = trainPerceptron(index, 3);
        y = sign(weights' * x);
        weights = weights + (eta * (d - y)) * x;
    end
end
end