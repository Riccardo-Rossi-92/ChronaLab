%% Test funzione Krakovska per Riccardo
clear; clc; 

% Creo sistema Rossler e faccio Embedding (dim=5,lag=5)
[~,Xlorenz] = CoupledLorenz(0,200,20000); 

embeddings = [5 10 20 30 40 50];
Dlorenz = zeros(2,2,length(embeddings)); % media e dev. std.

for i = 1:length(embeddings)

    Xlorenzemb = phaseSpaceReconstruction([Xlorenz(:,1) Xlorenz(:,4)],1,embeddings(i));
    % [D2,r_all(i,:),C(i,:)] = CorrelationDimension_TorVergata(Xlorenzemb);
    [Dlorenz(3,1,i),Dlorenz(3,2,i),~] = CorrelationDimension_MC(Xlorenzemb,1,"TorVergata");
    [Dlorenz(1,1,i),Dlorenz(1,2,i),~] = CorrelationDimension_MC(Xlorenzemb,1,"Krakosvka");
    [Dlorenz(2,1,i),Dlorenz(2,2,i),~] = CorrelationDimension_MC(Xlorenzemb,1,"Procaccia");
    % [Dlorenz(3,1,i),Dlorenz(3,2,i),~] = CorrelationDimension_MC(Xlorenzemb,1,"TorVergata");
    
end

%% Grafico

figure(1)

hold on
errorbar(embeddings,squeeze(Dlorenz(1,1,:)),squeeze(Dlorenz(1,2,:)),'LineWidth',1.5);
errorbar(embeddings,squeeze(Dlorenz(2,1,:)),squeeze(Dlorenz(2,2,:)),'LineWidth',1.5);
errorbar(embeddings,squeeze(Dlorenz(3,1,:)),squeeze(Dlorenz(3,2,:)),'LineWidth',1.5);

xlabel('Embedding dimension','Interpreter','latex','FontSize',10);
ylabel('$D_2$','Interpreter','latex','FontSize',10);

legend({"Krakovska","Procaccia","Tor Vergata"});