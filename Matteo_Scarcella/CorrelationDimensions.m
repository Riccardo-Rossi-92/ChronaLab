%% Calcola D2/ID per ciascun sistema
clear; clc; close;

% Creo sistemi e faccio Embedding
[~,Xlorenz] = CoupledLorenz(0,200,20000);
Xlorenz = phaseSpaceReconstruction(Xlorenz(:,1),5,5);

[Xrossler,~] = CoupledRossler2(0,500,50000); 
Xrossler = phaseSpaceReconstruction(Xrossler(:,1),5,5);

[~,Xlorenzuncoupled] = CoupledLorenz(0,200,20000);
Xlorenzuncoupled = phaseSpaceReconstruction([Xlorenzuncoupled(:,1) Xlorenzuncoupled(:,4)],5,5);

[~,Xlorenzcoupled] = CoupledLorenz(3.5,200,20000);
Xlorenzcoupled = phaseSpaceReconstruction([Xlorenzcoupled(:,1) Xlorenzcoupled(:,4)],5,5);

methods = ["Procaccia", "Krakosvka", "Autoencoder"];

Dlorenz = zeros(3,2);
Drossler = zeros(3,2);
Dlorenzuncoupled = zeros(3,2);
Dlorenzcoupled = zeros(3,2);

Loss = zeros(10,4);
% load Matteo_Scarcella/Models/Losses.mat

for i = 1:2
    
    [Dlorenz(i,1),Dlorenz(i,2),Loss(:,1)] = CorrelationDimension_MC(Xlorenz,50,methods(i));
    [Drossler(i,1),Drossler(i,2),Loss(:,2)] = CorrelationDimension_MC(Xrossler,50,methods(i));
    [Dlorenzuncoupled(i,1),Dlorenzuncoupled(i,2),Loss(:,3)] = CorrelationDimension_MC(Xlorenzuncoupled,50,methods(i));
    [Dlorenzcoupled(i,1),Dlorenzcoupled(i,2),Loss(:,4)] = CorrelationDimension_MC(Xlorenzcoupled,50,methods(i));

end

%% Calcola D2/ID per ciascun sistema al variare del rumore
clear; close; clc;

% carico i dati
[~,Xlorenz] = CoupledLorenz(0,200,20000);
[Xrossler,~] = CoupledRossler2(0,500,50000); 
[~,Xlorenzuncoupled] = CoupledLorenz(0,200,20000);
[~,Xlorenzcoupled] = CoupledLorenz(3.5,200,20000);

NoiseValues = [0.01,0.1,1];
methods = ["Procaccia", "Krakosvka", "Autoencoder"];

% load Matteo_Scarcella/Models/LossNoise.mat
LossNoise = zeros(10,4,3);

Drossler = zeros(3,2,3);
Dlorenz = zeros(3,2,3);
Dlorenzuncoupled = zeros(3,2,3);
DlorenzCoupled = zeros(3,2,3);

for i = 1:length(NoiseValues)

    disp("Noise value: " + NoiseValues(i));

    % Metto rumore sui dati
    Xlorenz_noise = normrnd(Xlorenz,NoiseValues(i));
    Xrossler_noise = normrnd(Xrossler,NoiseValues(i));
    Xlorenzuncoupled_noise = normrnd(Xlorenzuncoupled,NoiseValues(i));
    Xlorenzcoupled_noise = normrnd(Xlorenzcoupled,NoiseValues(i));

    % Embedding
    Xlorenz_noise = phaseSpaceReconstruction(Xlorenz_noise(:,1),5,5);
    Xrossler_noise = phaseSpaceReconstruction(Xrossler_noise(:,1),5,5);
    Xlorenzuncoupled_noise = phaseSpaceReconstruction([Xlorenzuncoupled_noise(:,1) Xlorenzuncoupled_noise(:,4)],5,5);
    Xlorenzcoupled_noise = phaseSpaceReconstruction([Xlorenzcoupled_noise(:,1) Xlorenzcoupled_noise(:,4)],5,5);
    
    for j = 1:3
        % Calculate the correlation dimension for the noisy data
        [Dlorenz(j,1,i),Dlorenz(j,2,i),LossNoise(:,1,i)] = CorrelationDimension_MC(Xlorenz_noise,50,methods(j));
        [Drossler(j,1,i),Drossler(j,2,i),LossNoise(:,2,i)] = CorrelationDimension_MC(Xrossler_noise,50,methods(j));
        [Dlorenzuncoupled(j,1,i),Dlorenzuncoupled(j,2,i),LossNoise(:,3,i)] = CorrelationDimension_MC(Xlorenzuncoupled_noise,50,methods(j));
        [Dlorenzcoupled(j,1,i),Dlorenzcoupled(j,2,i),LossNoise(:,4,i)] = CorrelationDimension_MC(Xlorenzcoupled_noise,50,methods(j));
    end

end