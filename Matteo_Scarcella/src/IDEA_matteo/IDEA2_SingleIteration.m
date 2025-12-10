function Loss = IDEA2_SingleIteration(X,EncoderStructure,CodeSize)
% Accetta una matrice X di dimensione MxN, dove M è il numero dei canali e
% N il numero di campioni.

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
[~,~,parameters] = IDEA_Network2(dlXn,0,[],VAE);
Results.Network.parameters = parameters;

% Model gradient
%accfun = dlaccelerate(@VAE_ModelGradient_Matteo);

% training options
VAE.LearningRate0 = 0.001;
VAE.DecayRate0 = 0.001; % più è basso, più è lento
VAE.MiniBatch = 500; 
VAE.MiniBatch = min(VAE.MiniBatch,N);
Saturation_checks_threshold = 10;
iteration = 0;
averageGrad = [];
averageSqGrad = [];

% Create plot
Loss_history = zeros(MaxEpochs ,1);

% figure(1);
% clf;
% hold on;
% set(gca,'YScale','log');
% h1 = plot(NaN, NaN, 'b', 'LineWidth', 2);
% xlabel("Epochs",'FontSize',10);
% ylabel("Loss",'FontSize',10);
% legend({'Loss (MSE)'});

% Addestramento
Loss = 0;
%LossesMeans = zeros(CodeSize);
%LossesStd = zeros(CodeSize);
%k = 1; % Indice dei neuroni CO
BestLoss = 100;
%BestLossLocal = 100;
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
        [gradients,~] = dlfeval(dlaccelerate(@IDEA_ModelGradient2),...
            parameters,VAE,dlXnBatch);

        % Learning rate
        LearningRate = VAE.LearningRate0./(1+VAE.DecayRate0*iteration);

        %ADAM
        [parameters,averageGrad,averageSqGrad] = adamupdate(parameters,gradients,averageGrad, ...
            averageSqGrad,iteration,LearningRate);
        
    end

    % disp(parameters.CO.weights);

    [~,Loss] = dlfeval(dlaccelerate(@IDEA_ModelGradient2),...
            parameters,VAE,dlXn);

    % Display plot
    Loss_history(epoch) = Loss;
   
    % set(h1, 'YData', Loss_history, 'XData', 1:numel(Loss_history));
    % drawnow limitrate;

    if Loss < 0.95*BestLoss
        Saturation_checks = 0;
        BestLoss = Loss;
    else
        Saturation_checks = Saturation_checks + 1;
    end
    
    % Per ogni neurone cerco il minimo valore di Loss raggiungibile 
    if Saturation_checks > Saturation_checks_threshold
        Loss = mean(Loss_history(epoch-Saturation_checks_threshold:epoch));
        % disp(LossesMeans(k));

        % parameters.CO.weights(k) = -0.001;

        % BestLoss = 100; % reset

        % Saturation_checks = 0;

        %k = k + 1;

        break;

    end
    
    % if isempty(parameters.CO.weights(parameters.CO.weights>0))
    %     break;
    % end

end



end