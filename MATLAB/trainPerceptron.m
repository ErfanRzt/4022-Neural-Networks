function [weights] = trainPerceptron(train, eta, w0, drawDecisionBoundry)
if nargin <= 3
    drawDecisionBoundry = true;
end

weights = w0';  % Initialize the Weights

trainSize = size(train);
iRandomInputs = randperm(trainSize(1));

figs = openfig('plotDatasets.fig');
plotTrainDataset = figs(1);

minx = min(train(:, 1));
maxx = max(train(:, 1));
miny = min(train(:, 2));
maxy = max(train(:, 2));
input = miny:0.1:maxy;

n = 0;
plots = [];
temp = weights;

% Activate the perceptron by applying input
% vector x(n) and desired response d(n)
for i = iRandomInputs
    n = n + 1;
    x = [1, train(i, 1:2)]';
    d = train(i, 3);
    y = sign(weights' * x);
    weights = weights + (eta * (d - y)) * x;

    if drawDecisionBoundry
        decisionBoundry = -(weights(1) + weights(3)*input) / weights(2);
        
        figure(plotTrainDataset);
        hBoundry = plot(decisionBoundry, input, 'k', 'LineWidth', 1.25);
        hold on; 
        hPoint = plot(train(i, 1), train(i, 2), 'mo', 'LineWidth', 4);
        xlim([1.2*minx, 1.2*maxx]);
        ylim([1.6*miny, 1.6*maxy]);
        grid on; axis equal;
        
        plots = [plots hBoundry];
        pause(0.05);
        
        delete(hPoint);
        if (temp ~= weights)
            delete(plots);
            temp = weights;
        end
    
        if (n == trainSize(1))
            input = 1.2*minx:0.1:1.2*maxx;
            decisionBoundry = -(weights(1) + weights(3)*input) / weights(2);
    
            hold on;
            hBoundry = plot(decisionBoundry, input, 'r-', 'LineWidth', 1.25);
        end
    end
end
end