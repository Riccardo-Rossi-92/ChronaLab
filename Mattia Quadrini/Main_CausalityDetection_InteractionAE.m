% Main_CausalityDetection_InteractionAE
% Causality detection using Interaction AutoEncoder approach

UseGPU = 0;

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
    clear all; clc;
    UseGPU = 0;
    disp("Using CPU environment")
end

%% Addpath

addpath src\autoencoder\
addpath src\generate_coupled_systems\
addpath src\embedding\

%% Dataset

% Here you can upload your data or generate a dataset for testing
% Some standard examples are uploaded here

Case = "Rossler-Rossler"; % Custom, Rossler-Rossler, ...

%%%%%%%%%%%%%%%%% Rossler Case %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if Case == "Rossler-Rossler"

    % generate coupled rossler system (see CoupledRossler for details)
    C = 0.6;
    [X,Y,t] = CoupledRossler(C);

    S = [X,Y];
    clear X Y
end

%% Embedding System X

embedding.case = 1;
embedding.variable = [1];
embedding.window = 10;
embedding.resolution = 1;
embedding.shifts = 2;
embedding.steps = 10;

if embedding.case == 1
    EmbX = Embedding_System(S,embedding);
    Xprev = EmbX{1}.E;
    Xnext = EmbX{2}.E;
end

N = size(Xnext,2);

%% Embedding System Y

embedding.case = 1;
embedding.variable = [4];
embedding.window = 10;
embedding.resolution = 1;
embedding.shifts = 2;
embedding.steps = 10;

if embedding.case == 1
    EmbY = Embedding_System(S,embedding);
    Yprev = EmbY{1}.E;
    Ynext = EmbY{2}.E;
end

%% deep learning array

dlXp = dlarray(Xprev,'CB');
dlXn = dlarray(Xnext,'CB');

dlYp = dlarray(Yprev,'CB');
dlYn = dlarray(Ynext,'CB');

dlXS = dlarray(std(Xnext,[],2),'CB');
dlYS = dlarray(std(Ynext,[],2),'CB');

if UseGPU == 1

    dlXp = gpuArray(dlXp);
    dlXn = gpuArray(dlXn);
    
    dlYp = gpuArray(dlYp);
    dlYn = gpuArray(dlYn);

    dlXS = gpuArray(dlXS);
    dlYS = gpuArray(dlYS);

end

%% Network Configuration and Initialisation

% Configuration

% Network architecture
AE.CodeSize = [2 2];
AE.Layer_En = [30 30 30];
AE.Layer_Dec = flip(AE.Layer_En);

AE.p_interaction_out = 0.1;

% training options
monitor_training = 1;

AE.LearningRate0 = 5e-3;
AE.DecayRate0 = 1e-3;

AE.MiniBatch = 1000;
AE.MiniBatch = min(AE.MiniBatch,N);

AE.iter_per_epochs = 100;
AE.max_epochs = 1000;

AE.Saturation_checks_threshold = 100;

% initialise and test
[~,~,~,~,parameters] = IAE_Network(dlXp(:,1:10),dlYp(:,1:10),0,[],AE);
[dlXnext,dlYnext,dlCX,dlCY] = IAE_Network(dlXp(:,1:51),dlYp(:,1:51),1,parameters,AE);

% Model gradient
accfun = dlaccelerate(@IAE_ModelGradient);

%% training initialisation

iteration = 0;
averageGrad = [];
averageSqGrad = [];

figure(1)
clf

BestLoss = 1;
Saturation_checks = 0;

for epoch = 1 :  AE.max_epochs
    tic
    for i = 1 : AE.iter_per_epochs

        % update iteration
        iteration = iteration + 1;

        % minibatch
        ind = randsample(N,AE.MiniBatch);

        dlX_n_now = dlXn(:,ind);
        dlX_p_now = dlXp(:,ind);

        dlY_n_now = dlYn(:,ind);
        dlY_p_now = dlYp(:,ind);

        % model gradient
        [gradients, Loss] = dlfeval(accfun,...
            parameters,AE,dlX_n_now,dlX_p_now,dlY_n_now,dlY_p_now,dlXS,dlYS);

        % Learning rate
        LearningRate = AE.LearningRate0./(1 + AE.DecayRate0*iteration);

        %ADAM
        [parameters,averageGrad,averageSqGrad] = adamupdate(parameters,gradients,averageGrad, ...
            averageSqGrad,iteration,LearningRate);

    end

    toc

    %% plot

    M_X_to_Y = double(extractdata(gather(parameters.Int.XY_w)));
    M_Y_to_X = double(extractdata(gather(parameters.Int.YX_w)));

    M_X_to_Y = sort(abs(M_X_to_Y(:)),'descend');
    M_Y_to_X = sort(abs(M_Y_to_X(:)),'descend');

    figure(1)
    subplot(2,2,1)
    plot(epoch,Loss,'.k','markersize',12)
    hold on
    grid on
    grid minor
    xlabel("epoch")
    ylabel("loss")
    set(gca,'yscale','log')

    subplot(2,4,3)
    hold off
    plot(M_X_to_Y,'.-b','markersize',16)
    hold on
    plot(M_Y_to_X,'.-r','markersize',16)
    grid on
    grid minor
    xlabel("Coeff #")
    ylabel("Matrix Coefficients")
    legend("M_{ X to Y}","M_{ Y to X}")

    drawnow

    %% error check

    check = 1;

    if check == 1

        AE_test = AE;
        AE_test.p_interaction_out = 0;
        [dlXpred,dlYpred] = IAE_Network(dlX_p_now,dlY_p_now,1,parameters,AE_test);

        err_X_with_Y = double(extractdata(gather(mean(dlXpred-dlX_n_now,1)))).^2;
        err_Y_with_X = double(extractdata(gather(mean(dlYpred-dlY_n_now,1)))).^2;

        AE_test.p_interaction_out = 1;
        [dlXpred,dlYpred] = IAE_Network(dlX_p_now,dlY_p_now,1,parameters,AE_test);

        err_X_alone = double(extractdata(gather(mean(dlXpred-dlX_n_now,1)))).^2;
        err_Y_alone = double(extractdata(gather(mean(dlYpred-dlY_n_now,1)))).^2;

        subplot(2,2,3)
        hold off
        histogram(log10(err_X_with_Y),"Normalization","pdf")
        hold on
        histogram(log10(err_X_alone),"Normalization","pdf")
        grid on
        grid minor
        xlabel("error X (log10)")
        ylabel("pdf")
        legend("With Y","Without Y")
        title("Prediction of X")

        subplot(2,2,4)
        hold off
        histogram(log10(err_Y_with_X),"Normalization","pdf")
        hold on
        histogram(log10(err_Y_alone),"Normalization","pdf")
        grid on
        grid minor
        xlabel("error Y (log10)")
        ylabel("pdf")
        legend("With X","Without X")
        title("Prediction of Y")

        %% Z_score - logE

        [h_Y_cause_X,p_Y_cause_X] = ttest2(log10(err_X_alone),log10(err_X_with_Y));
        [h_X_cause_Y,p_X_cause_Y] = ttest2(log10(err_Y_alone),log10(err_Y_with_X));

        if h_Y_cause_X == 1
            disp("Y causes X with p-value = " + p_Y_cause_X)
        else
            disp("Y may not cause X with p-value = " + p_Y_cause_X)
        end

        if h_X_cause_Y == 1
            disp("X causes Y with p-value = " + p_X_cause_Y)
        else
            disp("X may not cause Y with p-value = " + p_X_cause_Y)
        end

        % plot
        subplot(2,4,4)
        hold on
        plot(epoch,h_Y_cause_X,'.-b','markersize',16)
        plot(epoch,h_X_cause_Y,'o-r','markersize',8,'LineWidth',2)
        grid on
        grid minor
        xlabel("epoch")
        ylabel("decision (t-test  \alpha = 0.05")
        legend("h_{Y to X}","h_{X to Y}","Location","northwest")
        ylim([-0.1 1.1])

    end

    %% Saturation checks

    Loss = double(extractdata(gather(Loss)));
    
    if Loss < 0.95*BestLoss
        Results.parameters = parameters;
        Saturation_checks = 0;
        BestLoss = Loss;
    else
        Saturation_checks = Saturation_checks + 1;
    end

    if Saturation_checks > AE.Saturation_checks_threshold
        break
    end

    disp("Saturation Checks = " + Saturation_checks)

end


