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

%% Generate Data

[X,Y] = CoupledRossler(0.01);

%% time-series based configuration

embedding.case = 1;
embedding.variable = [1 1];
embedding.window = 5;

%% Embedding

if embedding.case == 1

    x = X(:,embedding.variable(1));
    y = Y(:,embedding.variable(2));

    Xt = buffer(x,embedding.window,embedding.window-1);
    Yt = buffer(y,embedding.window,embedding.window-1);

    Xnext = Xt(:,embedding.window+1:end);
    Xprev = Xt(:,1:end-embedding.window);

    Ynext = Yt(:,embedding.window+1:end);
    Yprev = Yt(:,1:end-embedding.window);

    clear Xt

elseif embedding.case == 0

    X = X';

    Xnext = X(:,2:end);
    Xprev = X(:,1:end-1);

end

%% deep learning array

dlXp = dlarray(Xprev,'CB');
dlXn = dlarray(Xnext,'CB');

dlYp = dlarray(Yprev,'CB');
dlYn = dlarray(Ynext,'CB');

dlXS = dlarray(std(Xnext,[],2),'CB');
dlYS = dlarray(std(Ynext,[],2),'CB');

%% Configuration

VAE.CodeSize = [2 2];

N = size(dlXn,2);

% Network architecture

VAE.p_interaction_out = 0.1;

VAE.Layer_En = [30 30];
VAE.Layer_Dec = flip(VAE.Layer_En);

[~,~,~,~,parameters] = InteractionAE(dlXp(:,1:10),dlYp(:,1:10),0,[],VAE);
[dlXpred,dlYpred,dlCX,dlCY] = InteractionAE(dlXp(:,1:10),dlYp(:,1:10),1,parameters,VAE);

% Model gradient
accfun = dlaccelerate(@InteractionAE_ModelGradient);

% training options
VAE.LearningRate0 = 5e-3;
VAE.DecayRate0 = 1e-3;

VAE.MiniBatch = 1000;
VAE.MiniBatch = min(VAE.MiniBatch,N);

% Saturation checks
Saturation_checks_threshold = 100;

%% training

iteration = 0;
averageGrad = [];
averageSqGrad = [];

figure(1)
clf

BestLoss = 1;
Saturation_checks = 0;

for epoch = 1 :  3000

    for i = 1 : 100

        % update iteration
        iteration = iteration + 1;

        % minibatch
        ind = randsample(N,VAE.MiniBatch);
        
        dlX_n_now = dlXn(:,ind);
        dlX_p_now = dlXp(:,ind);

        dlY_n_now = dlYn(:,ind);
        dlY_p_now = dlYp(:,ind);

        % model gradient
        [gradients, Loss] = dlfeval(accfun,...
            parameters,VAE,dlX_n_now,dlX_p_now,dlY_n_now,dlY_p_now,dlXS,dlYS);

        % Learning rate
        LearningRate = VAE.LearningRate0./(1+VAE.DecayRate0*iteration);

        %ADAM
        [parameters,averageGrad,averageSqGrad] = adamupdate(parameters,gradients,averageGrad, ...
            averageSqGrad,iteration,LearningRate);

    end

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
    ylabel("")
    set(gca,'yscale','log')

    subplot(2,2,2)
    hold off
    plot(M_X_to_Y,'.-b','markersize',16)
    hold on
    plot(M_Y_to_X,'.-r','markersize',16)

    drawnow

    %% error check

    check = 1;

    if check == 1
        
        VAE_test = VAE;
        VAE_test.p_interaction_out = 0;
        [dlXpred,dlYpred,dlCX,dlCY] = InteractionAE(dlX_p_now,dlY_p_now,1,parameters,VAE_test);

        err_X = double(extractdata(gather(mean(dlXpred-dlX_n_now,1))));
        err_Y = double(extractdata(gather(mean(dlYpred-dlY_n_now,1))));

        parameters_clean = parameters;
        parameters_clean.Int.XY_w = parameters_clean.Int.XY_w*0;
        parameters_clean.Int.YX_w = parameters_clean.Int.YX_w*0;

        [dlXpred,dlYpred,dlCX,dlCY] = InteractionAE(dlX_p_now,dlY_p_now,1,parameters_clean,VAE);

        err_X_clean = double(extractdata(gather(mean(dlXpred-dlX_n_now,1))));
        err_Y_clean = double(extractdata(gather(mean(dlYpred-dlY_n_now,1))));

        subplot(2,2,3)
        hold off
        histogram(err_X,"Normalization","pdf")
        hold on
        histogram(err_X_clean,"Normalization","pdf")

        subplot(2,2,4)
        hold off
        histogram(err_Y,"Normalization","pdf")
        hold on
        histogram(err_Y_clean,"Normalization","pdf")

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

    if Saturation_checks > Saturation_checks_threshold
        break
    end

    disp("Saturation Checks = " + Saturation_checks)

end

