%% Test funzione Krakovska per Riccardo
clear; clc; close;

% Creo sistema Rossler e faccio Embedding (dim=5,lag=5)
[~,Xlorenz] = CoupledLorenz(0,200,20000); 

embeddings = [5 10 20 30 40 50 60 70 80 90 100 200 500];
Dlorenz = zeros(2,2,length(embeddings)); % media e dev. std.

S = std(Xlorenz(:,1));

for i = 1:length(embeddings)

    Xlorenzemb = phaseSpaceReconstruction(Xlorenz(:,1),1,embeddings(i));

    Semb(i) = mean(std(Xlorenzemb,[],2));
    
end

figure(1)
clf
plot(embeddings,Semb./S)

