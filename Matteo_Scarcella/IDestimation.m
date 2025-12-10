%% Calcola ID per ciascun sistema
clear; clc; close;
% rng(1);

systems = ["Lorenz","Rossler","LorenzUncoupled","LorenzCoupled"];
methods = ["IDEA","TwoNN","IDEA-2"];

load Matteo_Scarcella/models/IDestimates.mat
% IDestimates = zeros(4,3);

for i = 1:4
    X = CreateSystem(systems(i)); % creo il sistema
    if systems(i) == "Lorenz" || systems(i) == "Rossler"
        X = phaseSpaceReconstruction(X(:,1),3,5);
    else
        X = phaseSpaceReconstruction([X(:,1) X(:,4)],3,5);
    end

    for j = 3:3
        IDestimates(i,j) = CorrelationDimension_MC(X,0,methods(j));
    end
end

%% Grafico 

for i = 1:4
    figure(i)
    clf 
    hold on
    plot(linspace(0,10,10),0.005*ones(10,1),'--','Color',[0.5 0.5 0.5]);
    plot(Losses(:,i),'-s','MarkerSize',5,'LineWidth',1.5);
   
    set(gca,'XMinorTick','off');
    set(gca,'YMinorTick','off');
    
    xlim([0.5 8.5]);
    
    axis square
    grid off
    
    xlabel('$\# Neuroni$','Interpreter','latex','FontSize',14);
    ylabel('$NMSE$','Interpreter','latex','FontSize',14);    
end

%% Calcola ID per ciascun sistema al variare di embedding*delay
clear; clc; close;

Dlorenz = zeros(3,9);
Drossler = zeros(3,9);
Dlorenzuncoupled = zeros(3,9);
Dlorenzcoupled = zeros(3,9);

% Loss = zeros(10,4);
% load Matteo_Scarcella/Models/Losses.mat

methods = ["IDEA","TwoNN","IDEA-2"];
embeddings = [5,10,20];
delays = [3,5,10];

for i = 2:3

    for j = 1:length(embeddings)*length(delays)
        
        [~,Xlorenz] = CoupledLorenz(0,100,10000);
        Xlorenz = phaseSpaceReconstruction(Xlorenz(:,1),delays(ceil(j/3)),embeddings(mod(j,3)+1));

        [Xrossler,~] = CoupledRossler2(0,100,20000); 
        Xrossler = phaseSpaceReconstruction(Xrossler(:,1),delays(ceil(j/3)),embeddings(mod(j,3)+1));

        [~,Xlorenzuncoupled] = CoupledLorenz(0,100,10000);
        Xlorenzuncoupled = phaseSpaceReconstruction([Xlorenzuncoupled(:,1) Xlorenzuncoupled(:,4)],delays(ceil(j/3)),embeddings(mod(j,3)+1));
        
        [~,Xlorenzcoupled] = CoupledLorenz(9,100,10000);
        Xlorenzcoupled = phaseSpaceReconstruction([Xlorenzcoupled(:,1) Xlorenzcoupled(:,4)],delays(ceil(j/3)),embeddings(mod(j,3)+1));
    
        Dlorenz(i,j) = CorrelationDimension_MC(Xlorenz,0,methods(i));
        Drossler(i,j) = CorrelationDimension_MC(Xrossler,0,methods(i));
        Dlorenzuncoupled(i,j) = CorrelationDimension_MC(Xlorenzuncoupled,0,methods(i));
        Dlorenzcoupled(i,j) = CorrelationDimension_MC(Xlorenzcoupled,0,methods(i));

    end
end

%% Grafico 

% x = sort(reshape(embeddings' * delays,[9,1]));
% for j = 1:length(embeddings)*length(delays)
% 
%     idx(j) = 
% end
% 
% figure(1);
% clf;
% hold on
% plot(x,Dlorenz(2,[1 3 7]));
% plot(x,Dlorenz(3,[2 5 8]));
% 
% xlabel('$\tau$','Interpreter','latex','FontSize',10);
% ylabel('$ID$','Interpreter','latex','FontSize',10);
% 
% legend({"TwoNN","IDEA-2"});

%% Calcola ID per ciascun sistema al variare di rumore
clear; clc; close;

systems = ["Lorenz","Rossler","LorenzUncoupled","LorenzCoupled"];
methods = ["IDEA","TwoNN","IDEA-2"];
NoiseValues = logspace(-2,0,5);

load Matteo_Scarcella/models/IDestimates_noise.mat
% IDestimates_noise = zeros(4,3,5);

for i = 1:4 % per ogni sistema

    for j = 2:3 % per ogni metodo

        for k = 1:length(NoiseValues) % per ogni valore di rumore

            X = CreateSystem(systems(i)); % creo il sistema
            X = normrnd(X,NoiseValues(k)); % applico rumore
            if systems(i) == "Lorenz" || systems(i) == "Rossler"
                X = phaseSpaceReconstruction(X(:,1),3,5);
            else
                X = phaseSpaceReconstruction([X(:,1) X(:,4)],3,5);
            end

            IDestimates_noise(i,j,k) = CorrelationDimension_MC(X,0,methods(j));
        end
    end
end

%% Grafico

load Matteo_Scarcella/models/IDestimates.mat
load Matteo_Scarcella/models/IDestimates_noise.mat

for i = 1:4
    figure(i)
    clf
    hold on
    plot(NoiseValues,squeeze(IDestimates_noise(i,1,:)),'-o','MarkerSize',10,'LineWidth',1.5);
    plot(NoiseValues,squeeze(IDestimates_noise(i,2,:)),'-o','MarkerSize',10,'LineWidth',1.5);
    plot(NoiseValues,squeeze(IDestimates_noise(i,3,:)),'-o','MarkerSize',10,'LineWidth',1.5);
    
    set(gca,"XScale","log");

    ticks = [0 5 10 15];
    ticks = [ticks IDestimates(i,3)];
    ticks = sort(ticks);
    yticks(ticks);
    % set(gca,'XMinorTick','off');
    % set(gca,'YMinorTick','off');
    
    ylim([0,15]);
    xlim([0.008 1.25]);
    
    axis square
    grid off
    
    xlabel('$\sigma$','Interpreter','latex','FontSize',14);
    ylabel('$ID$','Interpreter','latex','FontSize',14);
    
    legend({"IDEA","TwoNN","IDEA-2"});
end

