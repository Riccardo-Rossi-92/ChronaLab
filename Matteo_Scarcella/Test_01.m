clear; clc; close;

%% Generate system

C = 0.0;
[X,Y] = CoupledRossler(C);
Xcoupled = [X Y];

% ADD NOISE
% Xcoupled = normrnd(Xcoupled,0.1);
% il rumore ha dimensionalità massima quindi sovra-stima

%% time-series based configuration

embedding.case = 1;
embedding.variable = [1]; 
embedding.window = 5; % sceglie la dimensione della finestra di embedding

%% Embedding

if embedding.case == 1
    Xnext = [];
    Xprev = [];

    for i = 1 : length(embedding.variable)

        % prende la colonna i-esima (10001x1)
        x = Xcoupled(:,embedding.variable(i)); 
        
        % trasforma un vettore colonna in una matrice con embedding.window
        % righe e lenght(x) colonne.
        % ogni colonna è una finestra temporale che scorre in avanti dal
        % basso verso l'alto
        Xt = buffer(x,embedding.window,embedding.window-1);

        % scarta le prime 4 colonne, che contengono degli 0 di padding
        Xt = Xt(:,embedding.window:end);

        % prende gli ultimi 9992 valori
        Xnext = [Xnext; Xt(:,embedding.window+1:end)];

        % prende i primi 9992 valori
        Xprev = [Xprev; Xt(:,1:end-embedding.window)];
    end
    clear Xt

elseif embedding.case == 0

    X = X';

    Xnext = X(:,2:end);
    Xprev = X(:,1:end-1);

end

%% Calcola la dimensionalità intrinseca tramite il metodo Krakovska

D2 = CorrelationDimension_Krakosvka(Xnext');

%% Simulazione di un pendolo smorzato
clear; clc; close all;

% Parametri del pendolo
m = 0.1;              % massa [kg]
L = 1.5;              % lunghezza [m]
b = 0.25;            % coefficiente di smorzamento [N·m·s/rad]
g = 9.81;           % accelerazione gravitazionale [m/s^2]

theta0 = pi/2;      % angolo iniziale [rad]
omega0 = 0;         % velocità angolare iniziale [rad/s]

tspan = [0 50];     % intervallo di simulazione [s]
y0 = [theta0; omega0];

% Risoluzione numerica (ODE45)
pendolo = @(t, y) [y(2);
                   - (b/(m*L^2))*y(2) - (g/L)*sin(y(1))];

[t, Y] = ode45(pendolo, tspan, y0);

theta = Y(:,1);
omega = Y(:,2);

% --- Grafico del pendolo ---
figure('Name','Simulazione Pendolo Smorzato','NumberTitle','off')
subplot(2,1,1)
axis equal
hold on; grid on
xlim([-L-0.2 L+0.2])
ylim([-L-0.2 L+0.2])
title('Simulazione fisica del pendolo')

pivot = [0, 0];
h_line = plot([pivot(1), 0], [pivot(2), -L], 'LineWidth', 2);
h_mass = plot(0, -L, 'ro', 'MarkerFaceColor','r', 'MarkerSize', 10);

% --- Grafico di fase ---
subplot(2,1,2)
hold on; grid on
h_phase = plot(theta(1), omega(1), 'b', 'LineWidth', 1.5);
xlabel('\theta [rad]')
ylabel('\omega [rad/s]')
title('Phase plot')

% --- Animazione dinamica ---
for k = 1:length(t)
    % --- Animazione del pendolo ---
    subplot(2,1,1)
    x = L * sin(theta(k));
    y = -L * cos(theta(k));

    set(h_line, 'XData', [pivot(1) x], 'YData', [pivot(2) y]);
    set(h_mass, 'XData', x, 'YData', y);

    % --- Animazione del grafico di fase ---
    subplot(2,1,2)
    set(h_phase, 'XData', theta(1:k), 'YData', omega(1:k));
    xlim([-3*(1-0.003*k) 3*(1-0.003*k)]);
    ylim([-3*(1-0.003*k) 3*(1-0.003*k)]);
    
    drawnow
    pause(0.05)
end

disp('Simulazione completata.')

%% Simulazione dinamica preda-predatore
clear; clc; close all;

% Parametri del modello
alpha = 1.0;   % tasso di crescita delle prede
beta  = 0.5;   % tasso di predazione
delta = 0.3;   % efficienza dei predatori
gamma = 0.8;   % mortalità dei predatori

% Condizioni iniziali
x0 = 10;   % prede iniziali
y0 = 5;    % predatori iniziali
y_init = [x0; y0];

% Intervallo di simulazione
tspan = [0 30];

% Risoluzione numerica
LV = @(t, Y) [ alpha*Y(1) - beta*Y(1)*Y(2);
               delta*Y(1)*Y(2) - gamma*Y(2)];

[t, Y] = ode45(LV, tspan, y_init);

x = Y(:,1);  % prede
y = Y(:,2);  % predatori

% --- Grafico animato popolazioni ---
figure('Name','Lotka-Volterra','NumberTitle','off')

subplot(2,1,1)
hold on; grid on
plot_x = plot(t(1), x(1), 'b', 'LineWidth', 2);
plot_y = plot(t(1), y(1), 'r', 'LineWidth', 2);
xlabel('Tempo')
ylabel('Popolazione')
ylim([0 15]);
xlim([0 30]);
legend('Prede','Predatori')
title('Andamento delle popolazioni')

% --- Grafico di fase ---
subplot(2,1,2)
hold on; grid on
h_phase = plot(x(1), y(1), 'b', 'LineWidth', 1.5);
xlabel('Prede')
ylabel('Predatori')
ylim([0 10]);
xlim([0 15]);
title('Piano di fase')

% --- Animazione ---
subplot(2,1,1)
for k = 1:length(t)
    % Aggiorna grafico popolazioni
    set(plot_x, 'XData', t(1:k), 'YData', x(1:k));
    set(plot_y, 'XData', t(1:k), 'YData', y(1:k));
    
    % Aggiorna piano di fase
    subplot(2,1,2)
    set(h_phase, 'XData', x(max(k-30,1):k), 'YData', y(max(k-30,1):k));
    
    drawnow
    pause(0.05)
end

disp('Simulazione completata.')

%% Metodo prof.Murari

clear; close; clc
rng(30);

% Genero una superficie browniana (in teoria ID = 2.5)
H = 0.5;
n_points = 2^9;

[field,~,~,~] = Brownian_field(H,n_points);
field = field(1:end/2,1:end/2); % Estrae la sotto-matrice per rimuovere i NaN
dim = size(field,1);
tx = linspace(0,1,dim);
ty = linspace(0,1,dim);

% per calcolare la dimensione d2 con krakovska devo passare da uno
% spazio di matrice a uno spazio di vettore a 3 componenti (x,y,z).
X = zeros(dim*dim,3);
for h = 1:dim
    for k = 1:dim
        X((h-1)*dim+k,1) = tx(h);
        X((h-1)*dim+k,2) = ty(k);
        X((h-1)*dim+k,3) = field(h,k);
    end
end

plot3(X(:,1),X(:,2),X(:,3),'.');

D2_procaccia = correlationDimension(X);
D2_krakowska = CorrelationDimension_Krakosvka(X);

[D2,Loss] = AutoencoderCorrelationDimension(X',10,0.1);

figure(1);
plot(Loss);

disp(D2);
% 
% disp([D2_procaccia D2_krakowska]);
% 
% figure(1);
% surf(tx,ty,field,'EdgeColor','None');
% 
% % Ora, sapendo la dimensione target, applico due auto-encoder
% Xpred1 = Autoencoder(X',[100 50],2);
% Xpred2 = Autoencoder(X',[100 50],3);
% 
% % Faccio la differenza fra i due segnali e ne calcolo D2 tramite krakovska
% difference = Xpred2 - Xpred1;
% D2_difference = CorrelationDimension_Krakosvka(difference);
% 
% disp(D2_difference);

%% Metodo Riccardo

% L'idea è calcolare la dimensione D2 con metodo krakovska sul segnale
% uscente dal codice

clear; close; clc;

% Create data
X = CoupledRossler(0);
Xmean = mean(X(:,1));
Xstd = std(X(:,1));

% Embedding
embedding.variable = [1 2 3];
embedding.window = 5;
[Xnext,~] = Embedding(X,embedding.variable,embedding.window);

% Trasforma in deep learning array
dlXn = dlarray(Xnext,'CB');

% Network architecture
VAE.CodeSize = 2;
N = size(dlXn,2);
VAE.Layer_En = [100];
VAE.Layer_Dec = flip(VAE.Layer_En);
VAE.CodeSize = VAE.CodeSize;

% Inizializza la rete
[~,~,parameters] = VAE_Network_Matteo(dlXn,0,[],VAE);

% Model gradient
%accfun = dlaccelerate(@VAE_ModelGradient_Matteo);

% training options
VAE.LearningRate0 = 0.001;
VAE.DecayRate0 = 0.001; % più è basso, più è lento
VAE.MiniBatch = 500; 
VAE.MiniBatch = min(VAE.MiniBatch,N);
Saturation_checks_threshold = 25;
iteration = 0;
averageGrad = [];
averageSqGrad = [];

% Addestramento

BestLoss = 10;
Saturation_checks = 0;

for epoch = 1 : 5000

    % mescolo l'intero dataset
    idx = randperm(N); 

    for i = 1 : ceil(N / VAE.MiniBatch)

        % update iteration
        iteration = iteration + 1;

        % minibatch (in questo modo vedo tutti i campioni esattamente una
        % sola volta)
        startIdx = (i-1)*VAE.MiniBatch + 1;
        endIdx   = min(i*VAE.MiniBatch, N);  % evita di uscire dai limiti
        batchIdx = idx(startIdx:endIdx);
        
        dlXnBatch = dlXn(:,batchIdx);

        % model gradient
        [gradients, ~, ~, Loss] = dlfeval(dlaccelerate(@VAE_ModelGradient_Matteo),...
            parameters,VAE,dlXnBatch);

        % Learning rate
        LearningRate = VAE.LearningRate0./(1+VAE.DecayRate0*iteration);

        %ADAM
        [parameters,averageGrad,averageSqGrad] = adamupdate(parameters,gradients,averageGrad, ...
            averageSqGrad,iteration,LearningRate);
        
    end

    % Saturation checks
    Loss = double(extractdata(gather(Loss)));
    Results.Network.parameters = parameters;

    if Loss < 0.9*BestLoss % valore più alto -> criterio più rigido
        
        Saturation_checks = 0;
        BestLoss = Loss;
    else
        Saturation_checks = Saturation_checks + 1;
    end

    if Saturation_checks > Saturation_checks_threshold
        break
    end

end

% Predict 
[dlXpred,~] = VAE_Network_Matteo(dlXn,1,Results.Network.parameters,VAE,0);
[dlXpred_code,~] = VAE_Network_Matteo(dlXn,1,Results.Network.parameters,VAE,1);

% Extract data
Xpred_code = double(extractdata(gather(dlXpred_code)));
Xpred = double(extractdata(gather(dlXpred)));

% Calcola D2 (krakovska) per entrambi i segnali
% D2_normal = CorrelationDimension_Krakosvka(Xpred');
D2_code = CorrelationDimension_Krakosvka(Xpred_code');

% De-embedding
% Xpred_code = ReverseEmbedding(Xpred_code,2);
% Xpred = ReverseEmbedding(Xpred,embedding.window);

%% plot
figure(1);
hold on
plot( ((X(embedding.window*2:end,1) - Xmean) / Xstd) );
plot( ((Xpred - Xmean) / Xstd) );
plot( Xpred_code );

legend({'X','Xpred','Xpred (from code)'},'Location','northwest');

disp([D2_normal D2_code]);

%% Test Metodo IDEA

% Addestra una rete neurale di tipo auto-encoder ma con un pre-strato di
% codice latente, collegato con il codice latente attraverso una
% connessione di tipo uno a uno. Ciascun neurone del codice latente viene
% pesato, e la somma dei pesi viene minimizzata per trovare la ID

clear; close; clc;

% X = sobolset(6);
% X = X(1:10000,:);
% X = X';

% X = CoupledRossler(0);
% embedding.window = 3;
% embedding.variable = [1 2];
% [Xnext,~] = Embedding(X,embedding.variable,embedding.window);

% Genero una superficie browniana (in teoria ID = 2.5
% rng(2);
% H = 0.5;
% n_points = 2^9;
% [field,~,~,~] = Brownian_field(H,n_points);
% field = field(1:end/2,1:end/2); % Estrae la sotto-matrice per rimuovere i NaN
% dim = size(field,1);
% tx = linspace(0,1,dim);
% ty = linspace(0,1,dim);

% % per calcolare la dimensione d2 con krakovska devo passare da uno
% % spazio di matrice a uno spazio di vettore a 3 componenti (x,y,z).
% X = zeros(dim*dim,3);
% for h = 1:dim
%     for k = 1:dim
%         X((h-1)*dim+k,1) = tx(h);
%         X((h-1)*dim+k,2) = ty(k);
%         X((h-1)*dim+k,3) = field(h,k);
%     end
% end
% 
% X = X';
% disp(CorrelationDimension_Krakosvka(X));
% % plot3(X(:,1),X(:,2),X(:,3),'.');

[~,X] = LorenzAttractor(10,8/3,30,100);
embedding.window = 5;
embedding.variable = 1;
[Xnext,~] = Embedding(X,embedding.variable,embedding.window);
% Xnext = normrnd(Xnext,1);

% Trasforma in deep learning array
dlXn = dlarray(Xnext,'CB');

% Network architecture
VAE.CodeSize = 8;
N = size(dlXn,2);
VAE.Layer_En = [64 32 16];
VAE.Layer_Dec = flip(VAE.Layer_En);
VAE.CodeSize = VAE.CodeSize;

% Inizializza la rete
[~,~,parameters] = VAE_NetworkWeighted(dlXn,0,[],VAE);

BestCurrentWeight = 1;

% training options
VAE.LearningRate0 = 0.001;
VAE.DecayRate0 = 0; % più è basso, più è lento
VAE.MiniBatch = 500; 
VAE.MiniBatch = min(VAE.MiniBatch,N);
Saturation_checks_threshold = 100;
iteration = 0;
averageGrad = [];
averageSqGrad = [];

% Addestramento
BestLoss = 10;
Saturation_checks = 0;
MaxEpochs = 5000;

Loss_history = zeros(MaxEpochs ,1);
MSE_history  = zeros(MaxEpochs ,1);
Reg_history  = zeros(MaxEpochs ,1);

ActiveNeurons = VAE.CodeSize;

% Create plot
figure(1);
clf;
hold on;
set(gca,'YScale','log');
h1 = plot(NaN, NaN, 'b', 'LineWidth', 2);
h2 = plot(NaN, NaN, 'r', 'LineWidth', 2);
h3 = plot(NaN, NaN, 'g', 'LineWidth', 2);
legend({'Loss','MSE','Reg'});

for epoch = 1 : MaxEpochs 

    % mescolo l'intero dataset
    idx = randperm(N); 

    for i = 1 : ceil(N / VAE.MiniBatch)

        % update iteration
        iteration = iteration + 1;

        % minibatch (in questo modo vedo tutti i campioni esattamente una
        % sola volta)
        startIdx = (i-1)*VAE.MiniBatch + 1;
        endIdx   = min(i*VAE.MiniBatch, N);  % evita di uscire dai limiti
        batchIdx = idx(startIdx:endIdx);
        
        dlXnBatch = dlXn(:,batchIdx);

        % model gradient
        [gradients, MSE, Reg, Loss] = dlfeval(dlaccelerate(@VAE_ModelGradientWeighted),...
            parameters,VAE,dlXnBatch);

        % Learning rate
        LearningRate = VAE.LearningRate0./(1+VAE.DecayRate0*iteration);

        %ADAM
        [parameters,averageGrad,averageSqGrad] = adamupdate(parameters,gradients,averageGrad, ...
            averageSqGrad,iteration,LearningRate);       
    end

    % Criterio di arresto
    Loss_history(epoch) = Loss;
    MSE_history(epoch) = MSE;
    Reg_history(epoch) = Reg;
    
    set(h1, 'YData', Loss_history   , 'XData', 1:numel(Loss_history));
    set(h2, 'YData', MSE_history,  'XData', 1:numel(MSE_history));
    set(h3, 'YData', Reg_history,  'XData', 1:numel(Reg_history));
    drawnow limitrate;
    
    Weights = parameters.CO1.weights;
    
    % Se ho disattivato un neurone, aggiorno il numero di neuroni attivi e
    % riparto da zero 
    if sum(Weights>0) < ActiveNeurons
        ActiveNeurons = sum(Weights>0);
        BestCurrentWeight = 1;
        BestLoss = Loss;
    end
    
    % Assegno il peso corrente
    CurrentWeight = min(Weights(Weights>0));    

    if Loss < 1*BestLoss
        Results.Network.parameters = parameters;
        Saturation_checks = 0;
        BestLoss = Loss;
    else
        Saturation_checks = Saturation_checks + 1;
    end

    if Saturation_checks > Saturation_checks_threshold
        break
    end
    
    if (mod(epoch,5) == 0)
        disp(['---' newline ...
            'Epoca: ' num2str(epoch) newline ...
            'Loss: ' num2str(Loss) newline ...
            'Neuroni attivi: ' num2str(ActiveNeurons) newline ...
            'Somma dei pesi attivi: ' num2str(sum(Weights(Weights>0))) newline ...
            'Ultimo peso attivo: ' num2str(CurrentWeight) newline ...
            'Saturation Checks: ' num2str(Saturation_checks) newline ...
            '---']);
    end
end

%% Predict
[Xpred,LatentCode,~] = VAE_NetworkWeighted(dlXn,1,Results.Network.parameters,VAE);

% Extract data
LatentCode = double(extractdata(gather(LatentCode)));

[coeff, score, latent, tsquared, explained] = pca(LatentCode);

%%
var_cum = cumsum(explained);

target_var = 99.99; % percentuale di varianza che vuoi considerare

x = 1:length(explained);
y = cumsum(explained);
vx = 1:0.001:length(explained);
vq = interp1(x,y,vx,'linear');

ID_est = vx(find(vq >= target_var,1));

disp(ID_est);
plot(vx,vq);

%% Provo invece un metodo manuale, riducendo manualmente i neuroni del code

clear; close; clc;

X = LorenzAttractor(10,8/3,30,100);

embedding.case = 1;
embedding.variable = 1;
embedding.window = 5;
[D2,Loss] = AutoencoderCorrelationDimension(X',5,0.1,embedding);

figure(1);
clf;
normLoss = (Loss - min(Loss)) ./ (max(Loss)-min(Loss));
plot(normLoss);

%% Test autoencoder on rossler
clear;close;clc;

t_end = 1000;
[X,~] = CoupledRossler2(0,t_end,t_end*20);
X = normrnd(X,1);

embedding.window = 20;
Xnext = Embedding(X,1,embedding.window);

Xpred = Autoencoder(Xnext,[32 16],3);

Xdenoise = ReverseEmbedding(Xpred,embedding.window,1);

figure(1);
clf;
hold on
plot(X(embedding.window*2:end,1));
plot(Xdenoise);

%% prova

MaxCodeSize = 8;
embedding.case = 1;
embedding.variable = 1;
embedding.window = 5;

[X,~] = CoupledRossler2(0);
X = X';

Loss = zeros(MaxCodeSize,1);

Xdefault = X;

% Finding the loss values
for CodeSize = MaxCodeSize:-1:1
    
    X = Xdefault;

    % Embedding
    if (embedding.case == 1)
        [X,~] = Embedding(X',embedding.variable,embedding.window);
    end

    Xpred = Autoencoder(X,[32 16],CodeSize);

    % De-Embedding
    if (embedding.case == 1)
        X = ReverseEmbedding(X,embedding.window);
        Xpred = ReverseEmbedding(Xpred,embedding.window);
    end

    Loss(CodeSize) = mean((Xpred-X).^2,"all");

    disp("Iteration completed - Code size: " + CodeSize);

end

%% Find the elbow of the loss (corresponding to the D2 value)
D2 = 1;
Loss = (Loss-min(Loss)) ./ (max(Loss)-min(Loss));
LossDerivative = gradient(Loss);

clf;
hold on;
plot(Loss);
plot(LossDerivative);

disp(find(abs(LossDerivative)<0.1,1));


%%

figure;
ax = axes;
axis off;
view(3); axis equal;
hold on

c = Cube([0 0 0], 1);

c.populateChildren();
for i = 1:length(c.children)
    c.children(i).populateChildren();
    for j = 1:length(c.children(i).children)
        c.children(i).children(j).populateChildren();
    end
end

leaves = getLeaves(c);
for i = 1:length(leaves)
    leaves(i).drawCube(ax, 'FaceColor', [0.5 0.5 0.5], 'FaceAlpha', 0.05);
end

function leaves = getLeaves(obj)
    % Se il cubo è foglia, torna solo lui
    if isempty(obj.children)
        leaves = obj;
        return;
    end

    % Altrimenti aggrega le foglie dei figli
    leaves = Cube.empty;
    for k = 1:numel(obj.children)
        leaves = [leaves; getLeaves(obj.children(k))];
    end
end

%% 
k=8117;
for i = 1:5000
    x = -0.5 + rand(1,3);
    for j = 1:length(leaves)
        if leaves(j).checkInCube(x)
            k = k + 1;
            % plot3(x(1),x(2),x(3),'.');
            X(k,:) = x;
            break;
        end
    end
end

%% TEST PER PCA
clear; close; clc;

hold on

[~,X] = CoupledLorenz(0,200,20000);
Xemb = phaseSpaceReconstruction([X(:,1) X(:,4)],5,5);
[coeff, score, latent, tsquared, explained] = pca(Xemb);
figure(1);
plot(rescale(cumsum(latent),0,1))

[~,X] = CoupledLorenz(2.7,200,20000);
Xemb = phaseSpaceReconstruction([X(:,1) X(:,4)],5,5);
[coeff, score, latent, tsquared, explained] = pca(Xemb);
figure(1);
plot(rescale(cumsum(latent),0,1))