% Denoising_AE_TS
% Denoising Autoencoder for Time Series
%
%
%

%%

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

%% Dataset

% Here you can upload your data or generate a dataset for testing
% Some standard examples are uploaded here

Case = "Rossler"; % Custom, Rossler, ...

%%%%%%%%%%%%%%%%% Rossler Case %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if Case == "Rossler"

    % generate coupled rossler system (see CoupledRossler for details)
    [X,~,t] = CoupledRossler(0);


end

%% Noise

% This section can be used to add noise if you want to test the algorithm

% Add noise to the generated Rossler system data
noiseLevel = 0.1; % Define the noise level
Xnoisy = X + noiseLevel * randn(size(X)); % Add Gaussian noise

%% Embedding

embedding.case = 1;
embedding.variable = [1];
embedding.window = 10;
embedding.resolution = 1;
embedding.shifts = 1;
embedding.steps = 1;

if embedding.case == 1
    Emb = Embedding_System(Xnoisy,embedding);
    Y = Emb{1}.E;
else
    Y = Xnoisy';
end

N = size(Y,2);

%% Prepare deep learning array (dlarray)

dlY = dlarray(Y,'CB');
dlS = dlarray(std(Y,[],2),'CB');

%% Network Configuration and Initialisation

% Configuration

% Network architecture
AE.CodeSize = 3;
AE.Layer_En = [30 30 30];
AE.Layer_Dec = flip(AE.Layer_En);

% training options
monitor_training = 1;

AE.LearningRate0 = 5e-3;
AE.DecayRate0 = 1e-3;

AE.MiniBatch = 1000;
AE.MiniBatch = min(AE.MiniBatch,N);

AE.iter_per_epochs = 100;
AE.max_epochs = 1000;

% Saturation checks
Saturation_checks_threshold = 200;

% Initialise AE
[~,~,parameters] = AE_Network(dlY(:,1:10),0,[],AE);

% Test AE
[dlXpred,dlCode] = AE_Network(dlY(:,1:10),1,parameters,AE);

% Model gradient
accfun = dlaccelerate(@AE_ModelGradient);

%% Training

% Training - parameters initialisation

iteration = 0;
averageGrad = [];
averageSqGrad = [];

BestLoss = 1;
Saturation_checks = 0;



figure(1); clf;

for epoch = 1 :  AE.max_epochs

    for i = 1 : AE.iter_per_epochs

        % update iteration
        iteration = iteration + 1;

        % minibatch
        ind = randsample(N,AE.MiniBatch);
        dlY_now = dlY(:,ind);

        % model gradient
        [gradients, MSE, Reg, Loss] = dlfeval(accfun,...
            parameters,AE,dlY,dlS);

        % Learning rate
        LearningRate = AE.LearningRate0./(1+AE.DecayRate0*iteration);

        %ADAM
        [parameters,averageGrad,averageSqGrad] = adamupdate(parameters,gradients,averageGrad, ...
            averageSqGrad,iteration,LearningRate);

    end

    %% Predict and Plot results

    if monitor_training == 1
        dlYp = VAE_Network(dlY,1,parameters,AE);

        figure(1)
        subplot(1,3,1)
        plot(epoch,Loss,'.k','markersize',12)
        hold on
        plot(epoch,MSE,'.b','markersize',12)
        plot(epoch,Reg,'.r','markersize',12)
        grid on
        grid minor
        xlabel("epoch")
        ylabel("Loss")
        set(gca,'yscale','log')
        legend("Loss","MSE","Reg")

        subplot(1,3,[2 3])
        hold off
        plot(t,Xnoisy(:,embedding.variable),'-b')
        hold on
        plot(t(Emb{1}.ind),dlYp(end,:),'-r')
        plot(t,X(:,embedding.variable),'-.k')
        grid on
        grid minor
        xlim([0.5*max(t) 0.55*max(t)])
        legend("Noised","Reconstructed","Original")

    end
    drawnow

    %% Saturation checks

    Loss = double(extractdata(gather(Loss)));

    if Loss < 0.95*BestLoss
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

%% Reconstruction with reverse embedding
dlYp = VAE_Network(dlY,1,Results.Network.parameters,AE);
Yp = double(extractdata(gather(dlYp)));
Yp_rev = Embedding_Reverse(Yp,embedding.window);

%% Calculate performances

Results.RMSE_AE_orig = rmse(Yp(end,:),X(Emb{1}.ind,embedding.variable)');
Results.RMSE_rev_orig = rmse(Yp_rev,X(Emb{1}.ind,embedding.variable));

Results.RMSE_AE_noisy = rmse(Yp(end,:),Xnoisy(Emb{1}.ind,embedding.variable)');
Results.RMSE_rev_noisy = rmse(Yp_rev,Xnoisy(Emb{1}.ind,embedding.variable));

Results.RMSE_orig_noisy = rmse(X(Emb{1}.ind,embedding.variable),Xnoisy(Emb{1}.ind,embedding.variable));

disp(Results)

%% plot

figure(2)
clf
plot(t,Xnoisy(:,embedding.variable),'-b')
hold on
plot(t(Emb{1}.ind),dlYp(end,:),'-r')
plot(t,X(:,embedding.variable),'-.k')
plot(t(Emb{1}.ind),Yp_rev,'-')
grid on
grid minor
xlim([0.5*max(t) 0.55*max(t)])
legend("Noised","Reconstructed","Original","Reversed")

