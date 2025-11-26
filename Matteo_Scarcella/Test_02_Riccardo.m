%% Test funzione Krakovska per Riccardo
clear; clc; close;

% Creo sistema Rossler e faccio Embedding (dim=5,lag=5)
[~,Xlorenz] = CoupledLorenz(0,200,20000);

delays = [1 2 5 10];
embeddings = [5 10 20 30 40 50];

MC = 10;

for j = 1:length(delays)
    for i = 1:length(embeddings)

        Xlorenzemb = phaseSpaceReconstruction(Xlorenz(:,1),delays(j),embeddings(i));
        [D_krak(i,j),S_krak(i,j)] = CorrelationDimension_MC(Xlorenzemb,MC,"Krakosvka");
        [D_proc(i,j),S_proc(i,j)] = CorrelationDimension_MC(Xlorenzemb,MC,"Procaccia");
        [D_TV(i,j),S_TV(i,j)] = CorrelationDimension_MC(Xlorenzemb,MC,"TorVergata");

    end
end

%% Grafico

D_true = 2.05;

figure(1)
clf
subplot(1,3,1)
yline(D_true)
hold on
grid on
grid minor
xlabel("Embedding Window")
ylabel("D")
title("Krakosvka")

subplot(1,3,2)
yline(D_true)
hold on
grid on
grid minor
xlabel("Embedding Window")
ylabel("D")
title("Procaccia")

subplot(1,3,3)
yline(D_true)
hold on
grid on
grid minor
xlabel("Embedding Window")
ylabel("D")
title("Tor Vergata")

for j = 1 : length(delays)
    subplot(1,3,1)
    errorbar(embeddings,D_krak(:,j),S_krak(:,j),'.-','MarkerSize',12,'LineWidth',1.2)
    hold on

    ylim([D_true*0.75 D_true*1.25])


    subplot(1,3,2)
    errorbar(embeddings,D_proc(:,j),S_proc(:,j),'.-','MarkerSize',12,'LineWidth',1.2)
    hold on
    ylim([D_true*0.75 D_true*1.25])

    subplot(1,3,3)
    errorbar(embeddings,D_TV(:,j),S_TV(:,j),'.-','MarkerSize',12,'LineWidth',1.2)
    hold on
    ylim([D_true*0.75 D_true*1.25])
end

legend("delay = " + delays)

figure(2)
clf
subplot(1,3,1)
yline(D_true)
hold on
grid on
grid minor
xlabel("Embedding Window * delay")
ylabel("D")
title("Krakosvka")

subplot(1,3,2)
yline(D_true)
hold on
grid on
grid minor
xlabel("Embedding Window * delay")
ylabel("D")
title("Procaccia")

subplot(1,3,3)
yline(D_true)
hold on
grid on
grid minor
xlabel("Embedding Window * delay")
ylabel("D")
title("Tor Vergata")

for j = 1 : length(delays)
    subplot(1,3,1)
    errorbar(embeddings*delays(j),D_krak(:,j),S_krak(:,j),'.-','MarkerSize',12,'LineWidth',1.2)

    ylim([D_true*0.75 D_true*1.25])


    subplot(1,3,2)
    errorbar(embeddings*delays(j),D_proc(:,j),S_proc(:,j),'.-','MarkerSize',12,'LineWidth',1.2)

    ylim([D_true*0.75 D_true*1.25])

    subplot(1,3,3)
    errorbar(embeddings*delays(j),D_TV(:,j),S_TV(:,j),'.-','MarkerSize',12,'LineWidth',1.2)

    ylim([D_true*0.75 D_true*1.25])
end

legend("delay = " + delays)