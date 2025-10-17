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

D2_procaccia = correlationDimension(X);
D2_krakowska = CorrelationDimension_Krakosvka(X);

disp([D2_procaccia D2_krakowska]);

figure(1);
surf(tx,ty,field,'EdgeColor','None');

% Ora, sapendo la dimensione target, applico due auto-encoder
Xpred1 = Autoencoder(X',[100 50],2);
Xpred2 = Autoencoder(X',[100 50],3);

% Faccio la differenza fra i due segnali e ne calcolo D2 tramite krakovska
difference = Xpred2 - Xpred1;
D2_difference = CorrelationDimension_Krakosvka(difference);

disp(D2_difference);



