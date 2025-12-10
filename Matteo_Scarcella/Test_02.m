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

%% Test torvergata in funzione lag e embedding
clear; close; clc;

[~,X] = CoupledLorenz(0,200,20000);
X = normrnd(X,0.1);
% embeddings = [1 3 5 10 15 20 30 40 50];
lags = [1 3 5 10 15 20 30 40 50];

for i = 1:length(lags)
    Xemb = phaseSpaceReconstruction(X(:,1),lags(i),5);
    D2(i) = CorrelationDimension_TorVergata(Xemb,"Standard");
end

plot(lags,D2);

%% Correlation dimension (classic Procaccia)
clear; close; clc;

[~,X] = CoupledLorenz(0,100,10000);
Xemb = phaseSpaceReconstruction(X(:,1),5,5);

N = size(Xemb,1);
maxradius = 1;
distances = 0:0.1:maxradius;
C = zeros(length(distances),1);

for r = 1:length(distances)
    disp(distances(r));
    for i = 1:N
        for j = 1:N
            if i == j
                continue;
            end

            if norm(Xemb(i,:)-Xemb(j,:),2) < distances(r)
                C(r) = C(r) + 1;
            end
        end
    end
    C(r) = C(r) / (N*(N-1));

end

log_r = log(distances(2:end));
log_C = log(C(2:end));

plot(log_r,log_C);

b=linsolve([log_r' ones(size(log_r'))],log_C);
disp(b(1));

%% Correlation dimension (Rivisitato)
clear; close; clc;

[~,X] = CoupledLorenz(0,200,20000);
Xemb = phaseSpaceReconstruction(X(:,1),5,5);

N = size(Xemb,1);
D = pdist(Xemb); % calcola tutte le distanze per ciascuna coppia di punti

minradius = 0.02;
maxradius = 1;
distances = minradius:0.01:maxradius;  

histogram = histcounts(D,distances);

C = cumsum(histogram) / numel(D);

log_r = log(distances(2:end));
log_C = log(C);

b = linsolve([log_r' ones(size(log_r'))],log_C');

D2 = b(1);
disp(D2);



%% Test torvergata
clear; close; clc;

[~,X] = CoupledLorenz(0,100,10000);
NoiseValues = logspace(-3,0,5);

for i = 1:length(NoiseValues)

    Xnoise = normrnd(X,NoiseValues(i));
    Xemb = phaseSpaceReconstruction([Xnoise(:,1) Xnoise(:,4)],5,5);

    % D2_krakovska(i) =  CorrelationDimension_Krakosvka(Xemb);
    D2_procaccia(i) = correlationDimension(Xemb,0,1);
    D2_torvergata(i) = CorrelationDimension_TorVergata(Xemb,"Standard",200);
    % D2_matteo(i) = CorrelationDimension_Matteo(Xemb,2);
    D2_img(i) = CorrelationDimension_IMG(Xemb);

end

hold on
% plot(NoiseValues,D2_krakovska);
plot(NoiseValues,D2_procaccia);
plot(NoiseValues,D2_torvergata)
% plot(NoiseValues,D2_matteo);
plot(NoiseValues,D2_img);

set(gca,'XScale',"log")

legend({"Procaccia","Torvergata","IMG"});

%% D2 al variare del coefficiente di accoppiamento
clear;close;clc;

CouplingFactors = linspace(0,50,1);

for i = 1:length(CouplingFactors)
    [~,X] = CoupledLorenz(CouplingFactors(i),100,10000);
    Xemb = phaseSpaceReconstruction([X(:,1) X(:,4)],5,5);
    
    D2_procaccia(i) = correlationDimension(Xemb,0,1);
    % D2_krakosvka(i) = CorrelationDimension_Krakosvka(Xemb);
    D2_torvergata(i) = CorrelationDimension_TorVergata(Xemb,'Standard',300);
    % D2_matteo(i) = CorrelationDimension_Matteo(Xemb,1);
    D2_img(i) = CorrelationDimension_IMG(Xemb);
end

hold on
plot(CouplingFactors,D2_procaccia);
% plot(CouplingFactors,D2_krakosvka);
plot(CouplingFactors,D2_torvergata);
% plot(CouplingFactors,D2_matteo);
plot(CouplingFactors,D2_img);


legend({"Procaccia","Torvergata","IMG"});
