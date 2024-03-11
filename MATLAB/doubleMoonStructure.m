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

train = [ class(1:iTrain,:),        ones(iTrain, 1)   ];
test =  [ class(iTrain+1:end,:),    ones(N-iTrain, 1) ];

magnitude = (r-width)*ones(N,1) + rand(N,1)*width;
phase = pi + rand(N,1)*pi;

class = [magnitude.*cos(phase) + xBias/2, magnitude.*sin(phase) + yBias/2];
class = (R * class')';

train = [ train;    class(1:iTrain,:),      -1*ones(iTrain, 1)   ];
test =  [ test;     class(iTrain+1:end,:),  -1*ones(N-iTrain, 1) ];

plotOffsetx = separationDistance/4 + (r + width)/4;
plotOffsety = (r + width)/3;
if drawPatterns
    figure('Name','Train Dataset');
    plot(train(1:iTrain, 1), train(1:iTrain, 2), '.');
    xlim([- r - xBias/2 - plotOffsetx, 2*r - width/2 - xBias/2 + plotOffsetx]); 
    ylim([-separationDistance/2 - r - plotOffsety, r + separationDistance/2 + plotOffsety]);
    hold on; grid on;
    plot(train(iTrain+1:end, 3), train(iTrain+1:end, 4), '.');
    title('Train Dataset');

    figure('Name','Test Dataset');
    plot(test(1:(N-iTrain), 1), test(1:(N-iTrain), 2), '.');
    xlim([- r - xBias/2 - plotOffsetx, 2*r - width/2 - xBias/2 + plotOffsetx]); 
    ylim([-separationDistance/2 - r - plotOffsety, r + separationDistance/2 + plotOffsety]);
    hold on; grid on;
    plot(test((N-iTrain+1):end, 3), test((N-iTrain+1):end, 4), '.');
    title('Test Dataset');
end

end