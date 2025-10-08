clear; clc; close;

% Utilizzo un auto-encoder per rimozione del rumore dal segnale e calcolare
% indirettamente una stima della dimensione intrinseca

%% GPU

if false 

UseGPU = 1;

if UseGPU == 1
    if canUseGPU == 1
        gpu = gpuDevice();
        reset(gpu)

        gpu = gpuDevice();
        disp(gpu)
        wait(gpu)

        clear all; clc;

        UseGPU = 1;
        disp("GPU environment selected")
    else
        clear; clc;
        UseGPU = 0;
        disp("No available GPU")
    end

else
    clear; clc;
    UseGPU = 0;
    disp("Using CPU environment")
end

end

%% Creazione dei dati

% Crea un sistema di Rossler
[X,~] = CoupledRossler(0);

% Applica embedding
embedding.variable = 1;
embedding.window = 60;
[Xnext,~] = Embedding(X,embedding.variable,embedding.window); % embedding su dati puliti

% Trasforma in deep learning array
dlXn = dlarray(Xnext,'CB');

%% Configurazione rete neurale

% Network architecture
VAE.CodeSize = 16;
N = size(dlXn,2);
VAE.Layer_En = [64 32];
VAE.Layer_Dec = flip(VAE.Layer_En);
VAE.CodeSize = VAE.CodeSize;

% Inizializza la rete
[~,~,parameters] = VAE_Network_Matteo(dlXn,0,[],VAE);

% Model gradient
%accfun = dlaccelerate(@VAE_ModelGradient_Matteo);

% training options
VAE.LearningRate0 = 0.001;
VAE.DecayRate0 = 0.001; % più è basso, più è lento
VAE.MiniBatch = 250; 
VAE.MiniBatch = min(VAE.MiniBatch,N);
Saturation_checks_threshold = 250;
iteration = 0;
averageGrad = [];
averageSqGrad = [];

%% Addestramento

figure(1)
clf

BestLoss = 10;
Saturation_checks = 0;

for epoch = 1 : 3000

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
        
        dlXnBatch_clean = dlXn(:,batchIdx);
        dlXnBatch_noise = dlXnBatch_clean + randn(size(dlXnBatch_clean));

        % model gradient
        [gradients, MSE, Reg, Loss] = dlfeval(@VAE_ModelGradient_Matteo,...
            parameters,VAE,dlXnBatch_noise,dlXnBatch_clean);

        % Learning rate
        LearningRate = VAE.LearningRate0./(1+VAE.DecayRate0*iteration);

        %ADAM
        [parameters,averageGrad,averageSqGrad] = adamupdate(parameters,gradients,averageGrad, ...
            averageSqGrad,iteration,LearningRate);
        
    end

    % plot (every 10 epochs)
    if (mod(epoch,10) == 0) 

        figure(1)
        hold on
        plot(epoch,Loss,'.b',"markerSize",15)
        grid on
        grid minor
        xlabel("epoch")
        ylabel("")
        set(gca,'yscale','log')
    
        drawnow

    end

    % Saturation checks
    Loss = double(extractdata(gather(Loss)));

    if Loss < 0.999*BestLoss % valore più alto -> criterio più rigido
        Results.Network.parameters = parameters;
        Saturation_checks = 0;
        BestLoss = Loss;
    else
        Saturation_checks = Saturation_checks + 1;
    end

    if Saturation_checks > Saturation_checks_threshold
        break
    end

    disp("Saturation Checks = " + Saturation_checks)

end

%% Plot

dlXn_noise = dlXn + randn(size(dlXn));
[dlXpred,dlCode] = VAE_Network_Matteo(dlXn_noise,1,Results.Network.parameters,VAE);

Xpred = double(extractdata(gather(dlXpred)));
Xclean = double(extractdata(gather(dlXn)));
Xnoise = double(extractdata(gather(dlXn_noise)));

figure(2)
clf
plot(Xnoise(1,:),'-b')
hold on
plot(Xclean(1,:),'-g')
plot(Xpred(1,:),'-r')

MSE_data = mean((Xnoise-Xclean).^2,'all');
MSE_AE = mean((Xpred-Xclean).^2,'all');

disp([MSE_data MSE_AE])




