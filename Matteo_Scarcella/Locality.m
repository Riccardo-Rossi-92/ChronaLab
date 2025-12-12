%% Test funzione Krakovska per Riccardo
clear; clc; close;

% Creo sistema Rossler e faccio Embedding (dim=5,lag=5)
[~,Xlorenz] = CoupledLorenz(0,100,10000); 

% embeddings = [5 10 20 30 40 50 60 70 80 90 100 200 500];
lags = linspace(1,50,50);

S = std(Xlorenz(:,1));

for i = 1:length(lags)

    Xlorenzemb = phaseSpaceReconstruction(Xlorenz(:,1),lags(i),5);
    Semb(i) = mean(std(Xlorenzemb,[],2)); % lungo le righec

end

figure(1)
clf
plot(lags,Semb./S,'LineWidth',1)

xlabel('$\tau$','Interpreter','latex','FontSize',10);
ylabel('$\mathcal{R}_1(\tau)$','Interpreter','latex','FontSize',10);
