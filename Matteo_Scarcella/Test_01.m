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
embedding.variable = [1 2 3 4 5 6]; 
embedding.window = 3; % sceglie la dimensione della finestra di embedding

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