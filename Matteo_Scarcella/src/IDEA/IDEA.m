function ID = IDEA(X,EncoderStructure,CodeSize)
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
[~,~,parameters] = IDEA_Network(dlXn,0,[],VAE);
Results.Network.parameters = parameters;

% Model gradient
%accfun = dlaccelerate(@VAE_ModelGradient_Matteo);

% training options
VAE.LearningRate0 = 0.001;
VAE.DecayRate0 = 0.001; % più è basso, più è lento
VAE.MiniBatch = 500; 
VAE.MiniBatch = min(VAE.MiniBatch,N);
Saturation_checks_threshold = 100;
iteration = 0;
averageGrad = [];
averageSqGrad = [];

% Create plot
Loss_history = zeros(MaxEpochs ,1);
MSE_history  = zeros(MaxEpochs ,1);
Rec_history  = zeros(MaxEpochs ,1);
Reg_history  = zeros(MaxEpochs ,1);
Ort_history  = zeros(MaxEpochs ,1);

figure(1);
clf;
hold on;
set(gca,'YScale','log');
% h1 = plot(NaN, NaN, 'b', 'LineWidth', 2);
% h2 = plot(NaN, NaN, 'r', 'LineWidth', 2);
h3 = plot(NaN, NaN, 'b', 'LineWidth', 2);
xlabel("Epochs",'FontSize',10);
ylabel("Loss",'FontSize',10);
% h4 = plot(NaN, NaN, 'c', 'LineWidth', 2);
% h5 = plot(NaN, NaN, 'k', 'LineWidth', 2);
% legend({'Loss','MSE','Reconstruction','L1 Regularization','Orthogonality'});

% Addestramento
BestRec = 10;
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
        [gradients, ~, ~, ~, ~, ~] = dlfeval(@IDEA_ModelGradient,...
            parameters,VAE,dlXnBatch);

        % Learning rate
        LearningRate = VAE.LearningRate0./(1+VAE.DecayRate0*iteration);

        %ADAM
        [parameters,averageGrad,averageSqGrad] = adamupdate(parameters,gradients,averageGrad, ...
            averageSqGrad,iteration,LearningRate);
        
    end
    
    % Calcolo ricostruzione su tutto il dataset
    [~, ~, ~, Lrec, ~, ~] = dlfeval(@IDEA_ModelGradient,...
            parameters,VAE,dlXn);

    disp([parameters.CO1.weights parameters.CO2.weights]);

    % Display plot
    % Loss_history(epoch) = Loss;
    % MSE_history(epoch) = MSE;
    Rec_history(epoch) = Lrec;
    % Reg_history(epoch) = Reg;
    % Ort_history(epoch) = Lort;
    
    % set(h1, 'YData', Loss_history, 'XData', 1:numel(Loss_history));
    % set(h2, 'YData', MSE_history,  'XData', 1:numel(MSE_history));
    set(h3, 'YData', Rec_history,  'XData', 1:numel(Rec_history));
    % set(h4, 'YData', Reg_history,  'XData', 1:numel(Reg_history));
    % set(h5, 'YData', Ort_history,  'XData', 1:numel(Ort_history));
    drawnow limitrate;

    % Stop criterion
    if epoch > 10
        if Lrec < 0.995*mean(Rec_history(epoch-10:epoch-1))
            Results.Network.parameters = parameters;
            Saturation_checks = 0;
            % BestRec = Lrec;
        else
            Saturation_checks = Saturation_checks + 1;
        end
    
        if Saturation_checks > Saturation_checks_threshold 
            ID = length(parameters.CO1.weights(parameters.CO1.weights>0));
            break;
        end
    end

end

% Predict
[dlXpred,LatentCode] = IDEA_Network(dlXn,1,Results.Network.parameters,VAE);

% Extract data
Xpred = double(extractdata(gather(dlXpred)));
LatentCode = double(extractdata(gather(LatentCode)));

% Compute Loss
Loss = mean((Xpred-X).^2,"all");

end