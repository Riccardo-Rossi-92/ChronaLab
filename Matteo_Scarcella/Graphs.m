%% Rossler phase plot

[X,~] = CoupledRossler2(0,300,300*100);

figure(1)
plot3(X(:,1),X(:,2),X(:,3),'LineWidth',1);
title('3D Phase Plot','FontSize',14)

% figure(2)
% for i = 1:3
%     subplot(3,1,i)
%     plot(X(1:1000,i), 'LineWidth', 1.5)
%     grid on
%     xlabel('t')
%     ylabel(['X' num2str(i)])
% end
% sgtitle('Traiettorie delle coordinate X','FontSize',14);


%% D2 al variare del rumore
clear; close; clc;

windows = [2,5,10];

[X,~] = CoupledRossler2(0,1000,1000*20);
    
noise_values = [0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10];
D2_values = zeros(length(noise_values),1);

figure(1);
hold on
for j = 1:length(windows)

    for i = 1:length(noise_values)
        Xnoise = normrnd(X,noise_values(i));
    
        [Xnext,~] = Embedding(Xnoise,1,windows(j)); 
    
        D2_values(i) = CorrelationDimension_Krakosvka(Xnext');
        
    end
    
    plot(noise_values,D2_values,'-','LineWidth',2);    

end

plot(noise_values,2.01*ones(length(noise_values)),'--','LineWidth',2,'Color',[0.5 0.5 0.5]);

title('D2 respect to 0 mean noise with different values of std dev','FontSize',14);
xlabel('$\sigma$','Interpreter','latex','FontSize',10);
ylabel('$D_2$','Interpreter','latex','FontSize',10);
    
set(gca,'XScale','log');
ylim([0 10]);
    
grid off;
    
legend({'Estimation (Embedding=2)','Estimation (Embedding=5)','Estimation (Embedding=10)','Target'},'Location','northwest');
    
savefig('Matteo_Scarcella/Figures/D2-noise.fig')

%% D2 al variare del rumore con denoising
clear; close; clc;

windows = [2,5,10];

[X,~] = CoupledRossler2(0,1000,1000*20);
    
noise_values = [0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10];
D2_values = zeros(length(noise_values),1);


for j = 1:length(windows)

    for i = 1:length(noise_values)

        Xnoise = normrnd(X,noise_values(i));
    
        [Xnext,~] = Embedding(Xnoise,1,windows(j)); 

        Xnext_pred = Autoencoder(Xnext,[32 16],3);

        Xpred = ReverseEmbedding(Xnext_pred,windows(j));

        Xnext_denoised = Embedding(Xpred,1,windows(j));
    
        D2_values(i) = CorrelationDimension_Krakosvka(Xnext_denoised');
        
    end

    figure(2);
    hold on
    
    plot(noise_values,D2_values,'-','LineWidth',2);   

    drawnow

end

plot(noise_values,2.01*ones(length(noise_values)),'--','LineWidth',2,'Color',[0.5 0.5 0.5]);

title('D2 respect to 0 mean noise with different values of std dev (after denoising)','FontSize',14);
xlabel('$\sigma$','Interpreter','latex','FontSize',10);
ylabel('$D_2$','Interpreter','latex','FontSize',10);
    
set(gca,'XScale','log');
ylim([0 10]);
    
grid off;
    
legend({'Estimation (Embedding=2)','Estimation (Embedding=5)','Estimation (Embedding=10)','Target'},'Location','northwest');
    
savefig('Matteo_Scarcella/Figures/D2denoised-noise.fig')

%% D2 al variare della finestra di campionamento
clear;close;clc;

SamplingWindows = [10,20,50,100,200,500,1000,2000];
Embeddings = [2,5,10];
Precisions = [10,50,100];
D2 = zeros(length(SamplingWindows),1);

for p = 1:length(Precisions)

    figure(p);
    clf;
    hold on 

    for j = 1:length(Embeddings)
    
        for i = 1:length(SamplingWindows)
    
            % Genera Rossler
            [X,~] = CoupledRossler2(0,SamplingWindows(i),SamplingWindows(i)*Precisions(p));
        
            % Embedding
            embedding.window = Embeddings(j);
            Xnext = Embedding(X,1,embedding.window);
        
            % Calcola D2
            D2(i) = CorrelationDimension_Krakosvka(Xnext');
        
        end

        plot(SamplingWindows,D2,'-','LineWidth',2);
    
    end    

    plot(SamplingWindows,2.01*ones(length(SamplingWindows)),'--','LineWidth',2,'Color',[0.5 0.5 0.5]);
    
    title(sprintf('D2 estimation per sampling size (Precision=t_{end}*%d)',Precisions(p)),'FontSize',14);
    xlabel('$t_{end}$','Interpreter','latex','FontSize',10);
    ylabel('$D_2$','Interpreter','latex','FontSize',10);
    
    set(gca,'XScale','log');
    ylim([0 80]);
    
    grid off;
    
    legend({'Estimation (Embedding=2)','Estimation (Embedding=5)','Estimation (Embedding=10)','Target'},'Location','northwest');
    
    path = sprintf('Matteo_Scarcella/Figures/D2Estimation-samplingsize(%d).fig',Precisions(p));
    savefig(path);

end

%% D2 al variare del numero di punti
clear;close;clc;

PointsNumber = [1000,2000,5000,10000,20000,50000,100000];
Embeddings = [2,5,10];
D2 = zeros(length(PointsNumber),1);

figure(1);
clf;
hold on

for j = 1:length(Embeddings)

    for i = 1:length(PointsNumber)

        % Genera Rossler
        [X,~] = CoupledRossler2(0,1000,PointsNumber(i));
    
        % Embedding
        embedding.window = Embeddings(j);
        Xnext = Embedding(X,1,embedding.window);
    
        % Calcola D2
        D2(i) = CorrelationDimension_Krakosvka(Xnext');
    
    end

    plot(PointsNumber,D2,'-','LineWidth',2);

end

plot(PointsNumber,2.01*ones(length(PointsNumber)),'--','LineWidth',2,'Color',[0.5 0.5 0.5]);

title('D2 estimation per points number','FontSize',14);
xlabel('$\# points$','Interpreter','latex','FontSize',10);
ylabel('$D_2$','Interpreter','latex','FontSize',10);

set(gca,'XScale','log');

grid off;

legend({'Estimation (Embedding=2)','Estimation (Embedding=5)','Estimation (Embedding=10)','Target'},'Location','northwest');

savefig('Matteo_Scarcella/Figures/D2Estimation-pointsnumber.fig')

%% D2 accoppiato al variare del numero di punti
clear;close;clc;

PointsNumber = [1000,2000,5000,10000,20000,50000,100000];
Embeddings = [2,5,10];
D2 = zeros(length(PointsNumber),1);

figure(1);
clf;
hold on

for j = 1:length(Embeddings)

    for i = 1:length(PointsNumber)

        % Genera Rossler
        [X,Y] = CoupledRossler2(1,1000,PointsNumber(i));
    
        % Embedding
        embedding.window = Embeddings(j);
        Xnext = Embedding([X Y],[1 4],embedding.window);
    
        % Calcola D2
        D2(i) = CorrelationDimension_Krakosvka(Xnext');
    
    end

    plot(PointsNumber,D2,'-','LineWidth',2);

end

plot(PointsNumber,2.01*ones(length(PointsNumber)),'--','LineWidth',2,'Color',[0.5 0.5 0.5]);

title('D2 estimation of coupled system per points number','FontSize',14);
xlabel('$\# points$','Interpreter','latex','FontSize',10);
ylabel('$D_2$','Interpreter','latex','FontSize',10);

set(gca,'XScale','log');

grid off;

legend({'Estimation (Embedding=2)','Estimation (Embedding=5)','Estimation (Embedding=10)','Target'},'Location','northwest');

savefig('Matteo_Scarcella/Figures/D2EstimationCoupled-pointsnumber.fig')

%% D2 disaccoppiato per numero di punti
clear;close;clc;

PointsNumber = [1000,2000,5000,10000,20000,50000,100000];
Embeddings = [2,5,10];
D2 = zeros(length(PointsNumber),1);

figure(1);
clf;
hold on

for j = 1:length(Embeddings)

    for i = 1:length(PointsNumber)

        % Genera Rossler
        [X,Y] = CoupledRossler2(0,1000,PointsNumber(i));
    
        % Embedding
        embedding.window = Embeddings(j);
        Xnext = Embedding([X Y],[1 4],embedding.window);
    
        % Calcola D2
        D2(i) = CorrelationDimension_Krakosvka(Xnext');
    
    end

    plot(PointsNumber,D2,'-','LineWidth',2);

end

plot(PointsNumber,2.01*ones(length(PointsNumber)),'--','LineWidth',2,'Color',[0.5 0.5 0.5]);

title('D2 estimation of uncoupled system per points number','FontSize',14);
xlabel('$\# points$','Interpreter','latex','FontSize',10);
ylabel('$D_2$','Interpreter','latex','FontSize',10);

set(gca,'XScale','log');

grid off;

legend({'Estimation (Embedding=2)','Estimation (Embedding=5)','Estimation (Embedding=10)','Target'},'Location','northwest');

savefig('Matteo_Scarcella/Figures/D2EstimationUncoupled-pointsnumber.fig')



%% Causalità tramite D2(krakovska) al variare del coeff. di accoppiamento
% Dalle precedenti analisi, si ricavano i valori ritenuti ottimali per la
% stima della D2 in questo specifico sistema, che sono:
% - t_end > 500
% - points_number ~ 10000 (circa t_end*20)
% - embedding_size = 2-3

clear;close;clc;

CouplingFactors = [0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2];
NoiseValues = [1e-3,1e-2,1e-1,1e0];
Causality = zeros(length(CouplingFactors),length(NoiseValues));

for i = 1:length(CouplingFactors)

    for j = 1:length(NoiseValues)

        % Genera Rossler
        t_end = 1000;
        [X,Y] = CoupledRossler2(CouplingFactors(i),t_end,t_end*20);
        X = normrnd(X,NoiseValues(j));
        Y = normrnd(Y,NoiseValues(j));

        % Embedding
        embedding.window = 2;
        Xnext = Embedding(X,1,embedding.window);
        Ynext = Embedding(Y,1,embedding.window);
        XYnext = Embedding([X Y],[1 4],embedding.window);   

        % Calcola D2
        D2X = CorrelationDimension_Krakosvka(Xnext');
        D2Y = CorrelationDimension_Krakosvka(Ynext');
        D2XY = CorrelationDimension_Krakosvka(XYnext');
    
        Causality(i,j) = (D2X + D2Y) / D2XY;
    
    end
end

figure(1);
clf;
hold on

plot(CouplingFactors,Causality,'-','LineWidth',2);

title('Causality estimation respect to the coupling factor','FontSize',14);
xlabel('Coupling factor','FontSize',10);
ylabel('$\frac{D_2(X)+D_2(Y)}{D_2(X,Y)}$','Interpreter','latex','FontSize',10);

grid off;

legend({'\sigma = 10^{-3}','\sigma = 10^{-2}','\sigma = 10^{-1}','\sigma = 10^0'},'Location','northwest');

savefig('Matteo_Scarcella/Figures/CausalityEstimationD2.fig')

%% Causalità tramite D2(krakovska) al variare del coeff. di accoppiamento con denoising
% Dalle precedenti analisi, si ricavano i valori ritenuti ottimali per la
% stima della D2 in questo specifico sistema, che sono:
% - t_end > 500
% - points_number ~ 10000 (circa t_end*20)
% - embedding_size = 2-3

clear;close;clc;

CouplingFactors = [0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2];
NoiseValues = [1e-3,1e-2,1e-1,1e0];
Causality = zeros(length(CouplingFactors),length(NoiseValues));

for i = 1:length(CouplingFactors)

    for j = 1:length(NoiseValues)

        % Genera Rossler
        t_end = 1000;
        [X,Y] = CoupledRossler2(CouplingFactors(i),t_end,t_end*20);
        X = normrnd(X,NoiseValues(j));
        Y = normrnd(Y,NoiseValues(j));

        % Embedding
        embedding.window = 20;
        Xnext = Embedding(X,1,embedding.window);
        Ynext = Embedding(Y,1,embedding.window);
        XYnext = Embedding([X Y],[1 4],embedding.window);   

        Xpred = Autoencoder(Xnext,[32 16],3);
        Ypred = Autoencoder(Ynext,[32 16],3);
        XYpred = Autoencoder(XYnext,[32 16],6);

        % De-Embedding
        Xdenoise = ReverseEmbedding(Xpred,embedding.window,1);
        Ydenoise = ReverseEmbedding(Ypred,embedding.window,1);
        XYdenoise = ReverseEmbedding(XYpred,embedding.window,2);

        % Re-Embedding
        XnextDenoise = Embedding(Xdenoise,1,2);
        YnextDenoise = Embedding(Ydenoise,1,2);
        XYnextDenoise = Embedding(XYdenoise,[1 2],2);

        % Calcola D2
        D2X = CorrelationDimension_Krakosvka(XnextDenoise');
        D2Y = CorrelationDimension_Krakosvka(YnextDenoise');
        D2XY = CorrelationDimension_Krakosvka(XYnextDenoise');
    
        Causality(i,j) = (D2X + D2Y) / D2XY;
    
    end
end

figure(1);
clf;
hold on

plot(CouplingFactors,Causality,'-','LineWidth',2);

title('Causality estimation respect to the coupling factor with denoising','FontSize',14);
xlabel('Coupling factor','FontSize',10);
ylabel('$\frac{D_2(X)+D_2(Y)}{D_2(X,Y)}$','Interpreter','latex','FontSize',10);

grid off;

legend({'\sigma = 10^{-3}','\sigma = 10^{-2}','\sigma = 10^{-1}','\sigma = 10^0'},'Location','northwest');

savefig('Matteo_Scarcella/Figures/CausalityEstimationD2denoising.fig')