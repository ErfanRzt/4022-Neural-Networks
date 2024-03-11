function [train, test] = doubleMoonStructure( radius, width, rotation, separationDistance, ...
                                              datasetSize, trainTestRatio, drawPatterns )
switch nargin
    case 4
        datasetSize = 2000;
        trainTestRatio = 0.75;
        drawPatterns = true;
    case 5
        trainTestRatio = 0.75;
        drawPatterns = true;
    case 6
        drawPatterns = true;
end

nTrain = cast(datasetSize * trainTestRatio, "uint16");
iTrain = cast(nTrain / 2, "uint16");
N = cast(datasetSize / 2, "uint16");

theta = rotation * pi / 180;
R = [   cos(theta),  -sin(theta) ; 
        sin(theta),  cos(theta)  ];

r = radius + width/2; 

xBias = r - (width/2);
yBias = -separationDistance;

magnitude = (r-width)*ones(N,1) + rand(N,1)*width;
phase = rand(N,1)*pi;

class = [magnitude.*cos(phase) - xBias/2, magnitude.*sin(phase) - yBias/2];
class = (R * class')';

% train and test datasets with desired values;
% d = 1 and d = -1 for each class respectively
train = [ class(1:iTrain,:),        ones(iTrain, 1)   ];
test =  [ class(iTrain+1:end,:),    ones(N-iTrain, 1) ];

magnitude = (r-width)*ones(N,1) + rand(N,1)*width;
phase = pi + rand(N,1)*pi;

class = [magnitude.*cos(phase) + xBias/2, magnitude.*sin(phase) + yBias/2];
class = (R * class')';

% train and test datasets with desired values;
% d = 1 and d = -1 for each class respectively
train = [ train;    class(1:iTrain,:),      -1*ones(iTrain, 1)   ];
test =  [ test;     class(iTrain+1:end,:),  -1*ones(N-iTrain, 1) ];

plotOffsetx = separationDistance/4 + (r + width)/4;
plotOffsety = (r + width)/3;

visible = 'off';
if drawPatterns
    visible = 'on';
end

h(1) = figure('Name','Train Dataset', 'visible', visible);
plot(train(1:iTrain, 1), train(1:iTrain, 2), '.');
xlim([- r - xBias/2 - plotOffsetx, 2*r - width/2 - xBias/2 + plotOffsetx]); 
ylim([-separationDistance/2 - r - plotOffsety, r + separationDistance/2 + plotOffsety]);
hold on; grid on; axis equal;
plot(train(iTrain+1:end, 1), train(iTrain+1:end, 2), '.');
title('Train Dataset');

h(2) = figure('Name','Test Dataset', 'visible', visible);
plot(test(1:(N-iTrain), 1), test(1:(N-iTrain), 2), '.');
xlim([- r - xBias/2 - plotOffsetx, 2*r - width/2 - xBias/2 + plotOffsetx]); 
ylim([-separationDistance/2 - r - plotOffsety, r + separationDistance/2 + plotOffsety]);
hold on; grid on; axis equal;
plot(test((N-iTrain+1):end, 1), test((N-iTrain+1):end, 2), '.');
title('Test Dataset');

savefig(h, 'plotDatasets.fig', 'compact');

end