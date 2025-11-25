%% Test funzione Krakovska per Riccardo
clear; clc; close;

% Creo sistema Rossler e faccio Embedding (dim=5,lag=5)
[~,Xlorenz] = CoupledLorenz(0,200,20000); 

embeddings = [3,5,7,10];
Dlorenz = zeros(2,2,length(embeddings)); % media e dev. std.

for i = 1:length(embeddings)

    Xlorenzemb = phaseSpaceReconstruction([Xlorenz(:,1) Xlorenz(:,4)],5,i);

    [Dlorenz(1,1,i),Dlorenz(1,2,i),~] = CorrelationDimension_MC(Xlorenzemb,50,"Krakosvka");
    [Dlorenz(2,1,i),Dlorenz(2,2,i),~] = CorrelationDimension_MC(Xlorenzemb,50,"Procaccia");

end

% Grafico

figure(1)
hold on
errorbar(embeddings,squeeze(Dlorenz(1,1,:)),squeeze(Dlorenz(1,2,:)),'LineWidth',1.5);
errorbar(embeddings,squeeze(Dlorenz(2,1,:)),squeeze(Dlorenz(2,2,:)),'LineWidth',1.5);

xlabel('Embedding dimension','Interpreter','latex','FontSize',10);
ylabel('$D_2$','Interpreter','latex','FontSize',10);

legend({"Krakovska","Procaccia"});