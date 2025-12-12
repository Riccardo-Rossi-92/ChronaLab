%% ESEMPIO CARICAMENTO DATI
% load('Rossler-Rossler_C0.01.mat','err_X_with_Y', 'err_X_alone', 'err_Y_with_X', 'err_Y_alone','C')
% 
% Zlog_Y_X=mean(log10(err_X_with_Y))./mean(log10(err_X_alone));
% Zlog_X_Y=mean(log10(err_Y_with_X))./mean(log10(err_Y_alone));
% 
% figure(1)
% hold on
% plot(C,Zlog_X_Y,'.b',"MarkerSize",16)
% plot(C,Zlog_Y_X,'.r',"MarkerSize",16)

%% GRAFICO INIZIALE
Cgraph=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12];
Zlog_Y_X_vect=zeros(length(Cgraph),1);
Zlog_X_Y_vect=zeros(length(Cgraph),1);

for i=1:length(Cgraph)
    filename = sprintf('RosslerLorenz_C%.2f.mat', Cgraph(i));
    load(filename,'err_X_with_Y', 'err_X_alone', 'err_Y_with_X', 'err_Y_alone','C')
    Zlog_Y_X_vect(i)=mean(log10(err_X_with_Y))./mean(log10(err_X_alone));
    Zlog_X_Y_vect(i)=mean(log10(err_Y_with_X))./mean(log10(err_Y_alone));

end

figure(1)
hold on
grid on

plot(Cgraph, Zlog_Y_X_vect, '.-b', 'MarkerSize', 16);   % linea + punti
plot(Cgraph, Zlog_X_Y_vect, '.-r', 'MarkerSize', 16);

xlabel('C')
ylabel('Z-values')
legend('Zlog Y/X', 'Zlog X/Y')

%%
clear all
clc
%% SECONDO GRAFICO

Cgraph=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12];
PredX=zeros(length(Cgraph),1);
PredY=zeros(length(Cgraph),1);

for i=1:length(Cgraph)
    filename = sprintf('RosslerLorenz_C%.2f.mat', Cgraph(i));
    load(filename,'err_X_with_Y', 'err_X_alone', 'err_Y_with_X', 'err_Y_alone','C')
    PredX(i)=mean(log10(err_X_alone))./mean(log10(err_X_with_Y))-1;
    PredY(i)=mean(log10(err_Y_alone))./mean(log10(err_Y_with_X))-1;

end

% Se negativo, la conoscenza ci migliora la stima
% Se positivo, la conoscenza porta a peggioramento(?)
% Se nullo, non influenza le prestazioni


figure(2)
hold on
grid on

plot(Cgraph, PredX, '.-b', 'MarkerSize', 16);   % linea + punti
plot(Cgraph, PredY, '.-r', 'MarkerSize', 16);

xlabel('C')
legend('PredX', 'PredY')