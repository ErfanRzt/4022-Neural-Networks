function [class1, class2, accuracy] = testPerceptron(test, weights, drawDecisionBoundry)
if nargin >= 2
        drawDecisionBoundry = true;
end

testClassSize = size(test);
testPerceptron = [ test(:,1:2)  ones(testClassSize(1),1)    zeros(testClassSize(1),1) ;
                   test(:,3:4)  -1*ones(testClassSize(1),1) zeros(testClassSize(1),1) ];
error = 0;
testSize = size(testPerceptron);
for i = 1:testSize(1)
    x = [1, testPerceptron(i, 1:2)]';
    d = testPerceptron(i, 3);
    y = sign(weights' * x);

    testPerceptron(i, 4) = y;

    if y ~= d
        error = error + 1;
        accuracy = 100 * ((testSize(1)-error) / testSize(1));
    end
end

m = 0;
n = 0;

class1 = testPerceptron;
class2 = testPerceptron;

for i = 1:testSize(1)
    if testPerceptron(i, 4) == 1
        class2(i-m, :) = [];
        m = m + 1;
    elseif testPerceptron(i, 4) == -1
        class1(i-n, :) = [];
        n = n + 1;
    end
end

if drawDecisionBoundry
    minx = min([min(class1(:, 1)), min(class2(:, 1))]);
    maxx = max([max(class1(:, 1)), max(class2(:, 1))]);
    miny = min([min(class1(:, 2)), min(class2(:, 2))]);
    maxy = max([max(class1(:, 2)), max(class2(:, 2))]);

    minx = 1.2*minx;
    maxx = 1.2*maxx;
    miny = 1.6*miny;
    maxy = 1.6*maxy;

    input = minx:0.1:maxx;
    decisionBoundry = -(weights(1) + weights(3)*input) / weights(2);
    
    figure('Name','Perceptron Test Results');
    plot(class1(:, 1), class1(:, 2), '.');
    hold on;
    plot(class2(:, 1), class2(:, 2), '.');
    hold on;
    plot(decisionBoundry, input, '--', 'LineWidth', 1.5);
    xlim([minx, maxx]);
    ylim([miny, maxy]);
    grid on;
    title('Perceptron Test Results');
end
end