function Xpred = Autoencoder(X,EncoderStructure,CodeSize)

% Trasforma in deep learning array
dlXn = dlarray(X,'CB');

% Network architecture
VAE.CodeSize = CodeSize;
N = size(dlXn,2);
VAE.Layer_En = EncoderStructure;
VAE.Layer_Dec = flip(VAE.Layer_En);
VAE.CodeSize = VAE.CodeSize;

% Inizializza la rete
[~,~,parameters] = VAE_Network_Matteo(dlXn,0,[],VAE);

% Model gradient
%accfun = dlaccelerate(@VAE_ModelGradient_Matteo);

% training options
VAE.LearningRate0 = 0.001;
VAE.DecayRate0 = 0.001; % più è basso, più è lento
VAE.MiniBatch = 500; 
VAE.MiniBatch = min(VAE.MiniBatch,N);
Saturation_checks_threshold = 25;
iteration = 0;
averageGrad = [];
averageSqGrad = [];

%% Addestramento

BestLoss = 10;
Saturation_checks = 0;

for epoch = 1 : 5000

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

    % Saturation checks
    Loss = double(extractdata(gather(Loss)));
    Results.Network.parameters = parameters;

    if Loss < 0.9*BestLoss % valore più alto -> criterio più rigido
        
        Saturation_checks = 0;
        BestLoss = Loss;
    else
        Saturation_checks = Saturation_checks + 1;
    end

    if Saturation_checks > Saturation_checks_threshold
        break
    end

end

% Predict
[dlXpred,~] = VAE_Network_Matteo(dlXn,1,Results.Network.parameters,VAE);

% Extract data
Xpred = double(extractdata(gather(dlXpred)));

end