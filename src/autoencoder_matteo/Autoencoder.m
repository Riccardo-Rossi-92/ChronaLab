function [Xpred,LatentCode] = Autoencoder(X,EncoderStructure,CodeSize)

% Trasforma in deep learning array
dlXn = dlarray(X,'CB');

% Network architecture
VAE.CodeSize = CodeSize;
N = size(dlXn,2);
VAE.Layer_En = EncoderStructure;
VAE.Layer_Dec = flip(VAE.Layer_En);
VAE.CodeSize = VAE.CodeSize;
MaxEpochs = 5000;

% Inizializza la rete
[~,~,parameters] = VAE_Network_Matteo(dlXn,0,[],VAE);
Results.Network.parameters = parameters;

% Model gradient
%accfun = dlaccelerate(@VAE_ModelGradient_Matteo);

% training options
VAE.LearningRate0 = 0.001;
VAE.DecayRate0 = 0.001; % più è basso, più è lento
VAE.MiniBatch = 500; 
VAE.MiniBatch = min(VAE.MiniBatch,N);
Saturation_checks_threshold = 200;
iteration = 0;
averageGrad = [];
averageSqGrad = [];

% Create plot
Loss_history = zeros(MaxEpochs ,1);
MSE_history  = zeros(MaxEpochs ,1);
Reg_history  = zeros(MaxEpochs ,1);

figure(1);
clf;
hold on;
set(gca,'YScale','log');
h1 = plot(NaN, NaN, 'b', 'LineWidth', 2);
h2 = plot(NaN, NaN, 'r', 'LineWidth', 2);
h3 = plot(NaN, NaN, 'g', 'LineWidth', 2);
legend({'Loss','MSE','Reg'});

% Addestramento
BestLoss = 10;
Saturation_checks = 0;

for epoch = 1 : MaxEpochs

    % mescolo l'intero dataset
    idx = randperm(N); 

    for i = 1 : ceil(N / VAE.MiniBatch)

        % update iteration
        iteration = iteration + 1;

        % minibatch (in questo modo vedo tutti i campioni esattamente una
        % sola volta)
        startIdx = (i-1)*VAE.MiniBatch + 1;
        endIdx   = min(i*VAE.MiniBatch, N);  % evita di uscire dai limiti
        batchIdx = idx(startIdx:endIdx);
        
        dlXnBatch = dlXn(:,batchIdx);

        % model gradient
        [gradients, ~, ~, Loss] = dlfeval(dlaccelerate(@VAE_ModelGradient_Matteo),...
            parameters,VAE,dlXnBatch);

        % Learning rate
        LearningRate = VAE.LearningRate0./(1+VAE.DecayRate0*iteration);

        %ADAM
        [parameters,averageGrad,averageSqGrad] = adamupdate(parameters,gradients,averageGrad, ...
            averageSqGrad,iteration,LearningRate);
        
    end

    % Display plot
    Loss_history(epoch) = Loss;
    % MSE_history(epoch) = MSE;
    % Reg_history(epoch) = Reg;
    
    set(h1, 'YData', Loss_history   , 'XData', 1:numel(Loss_history));
    % set(h2, 'YData', MSE_history,  'XData', 1:numel(MSE_history));
    % set(h3, 'YData', Reg_history,  'XData', 1:numel(Reg_history));
    drawnow limitrate;

    % Saturation checks
    Loss = double(extractdata(gather(Loss)));

    if Loss < 0.95*BestLoss
        Results.Network.parameters = parameters;
        Saturation_checks = 0;
        BestLoss = Loss;
    else
        Saturation_checks = Saturation_checks + 1;
    end

    if Loss < 1e-3 || Saturation_checks > Saturation_checks_threshold 
        break
    end

end

% Predict
[dlXpred,LatentCode] = VAE_Network_Matteo(dlXn,1,Results.Network.parameters,VAE);

% Extract data
Xpred = double(extractdata(gather(dlXpred)));
LatentCode = double(extractdata(gather(LatentCode)));


end