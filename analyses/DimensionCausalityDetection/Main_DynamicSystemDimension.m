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

[X,Y] = CoupledRossler(0.1);

% X = Y;
X = [X Y];

%% time-series based configuration

embedding.case = 1;
embedding.variable = [4];
embedding.window = 5;

%% Embedding

if embedding.case == 1
    Xnext = [];
    Xprev = [];

    for i = 1 : length(embedding.variable)

        x = X(:,embedding.variable(i));
        Xt = buffer(x,embedding.window,embedding.window-1);

        Xt = Xt(:,embedding.window:end);

        Xnext = [Xnext; Xt(:,embedding.window+1:end)];
        Xprev = [Xprev; Xt(:,1:end-embedding.window)];
    end
    clear Xt

elseif embedding.case == 0

    X = X';

    Xnext = X(:,2:end);
    Xprev = X(:,1:end-1);

end

%% deep learning array

dlXp = dlarray(Xprev,'CB');
dlXn = dlarray(Xnext,'CB');
dlS = dlarray(std(Xnext,[],2),'CB');

%% Configuration

VAE.CodeSizes = 1:6;

for l = 1 : length(VAE.CodeSizes)

    %% Flow Net Initialisation

    N = size(dlXn,2);

    % Network architecture

    VAE.Layer_En = [30 30 30 30];
    VAE.Layer_Dec = flip(VAE.Layer_En);
    VAE.CodeSize = VAE.CodeSizes(l);

    [~,~,parameters] = VAE_Network(dlXn(:,1:10),0,[],VAE);
    [dlXpred,dlCode] = VAE_Network(dlXn(:,1:10),1,parameters,VAE);

    % Model gradient
    accfun = dlaccelerate(@VAE_ModelGradient);

    % Weights to be used
    % VAE.lambdas = [1e-5 1e-4 1e-3 1e-2 1e-1 1];
    VAE.lambdas = [linspace(1e-4,1e-3,11) linspace(2e-3,1e-2,10) linspace(2e-2,1e-1,10) linspace(2e-1,1,10)];

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

            % model gradient
            [gradients, MSE, Reg, Loss] = dlfeval(accfun,...
                parameters,VAE,dlX_n_now,dlX_p_now,dlS);

            % Learning rate
            LearningRate = VAE.LearningRate0./(1+VAE.DecayRate0*iteration);

            %ADAM
            [parameters,averageGrad,averageSqGrad] = adamupdate(parameters,gradients,averageGrad, ...
                averageSqGrad,iteration,LearningRate);

        end

        % plot
        figure(1)
        plot(epoch,Loss,'.g','markersize',12)
        hold on
        plot(epoch,MSE,'.b','markersize',12)
        plot(epoch,Reg,'.r','markersize',12)
        grid on
        grid minor
        xlabel("epoch")
        ylabel("")
        set(gca,'yscale','log')

        drawnow

        %% Saturation checks
        Loss = double(extractdata(gather(Loss)));
        if Loss < 0.95*BestLoss
            Results.Network(l).parameters = parameters;
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

    %% store data
    Results.MSE(l) = double(extractdata(gather(MSE)));
    

    figure(2)
    hold on
    plot(VAE.CodeSizes(1:l),Results.MSE(1:l),'.-b','markersize',16)
    set(gca,'yscale','log')

end

