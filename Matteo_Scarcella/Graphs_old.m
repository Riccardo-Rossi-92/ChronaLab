clear; clc; close;

%% Plot Rossler Attractor

[X,~] = CoupledRossler(0);

figure(1)
plot3(X(1:1000,1),X(1:1000,2),X(1:1000,3),'LineWidth',1.5);
title('3D Phase Plot','FontSize',14)

figure(2)
for i = 1:3
    subplot(3,1,i)
    plot(X(1:1000,i), 'LineWidth', 1.5)
    grid on
    xlabel('t')
    ylabel(['X' num2str(i)])
end
sgtitle('Traiettorie delle coordinate X','FontSize',14);



%% D2 al variare del rumore

n = 20;
embedding.variable = 1;
windows = [3,5,10,20];
colors = ['r','g','b','c'];

[X,~] = CoupledRossler(0);

figure(3);

for emb = 1:length(windows)
    
    embedding.window = windows(emb);

    noise_values = linspace(1e-3,1e0,n);
    D2_values = zeros(n,1);
    
    for i = 1:n
        Xnoise = normrnd(X,noise_values(i));

        [Xnext,~] = Embedding(Xnoise,embedding.variable,embedding.window); 

        D2_values(i) = CorrelationDimension_Krakosvka(Xnext');

        disp("Iteration " + i);
    end

    hold on
    plot(noise_values,D2_values,'Color',colors(emb));

    drawnow
    
    % yline(embedding.window,'--','Color',[0.5 0.5 0.5],'LabelHorizontalAlignment','left');
    
end

title('D2 dimension respect to the noise','FontSize',14);
xlabel('Noise level','FontSize',10);
ylabel('D2 dimension','FontSize',10);

set(gca,'XScale','log');
ylim([0 14]);
grid off;

legend({'embedding.window = 3','embedding.window = 5','embedding.window = 10','embedding.window = 20'},'Location','northwest');

savefig('Matteo_Scarcella/Figures/D2-noise.fig')

%% D2 al variare del rumore per segnale denoised

n = 20;
embedding.variable = 1;
windows = [3,5,10,20];
colors = ['r','g','b','c'];

[X,~] = CoupledRossler(0);

figure(4);

for emb = 1:length(windows)
    
    embedding.window = windows(emb);

    noise_values = linspace(1e-3,1e0,n);
    D2_values = zeros(n,1);
    % MSE_values = zeros(n,1);
    
    for i = 1:n
        
        % Applica il rumore
        Xnoise = normrnd(X,noise_values(i));
        
        % Applica l'embedding
        [Xnext,~] = Embedding(Xnoise,embedding.variable,embedding.window); 
        
        % Rimuove il rumore tramite auto-encoder
        Xnext_pred = Autoencoder(Xnext);

        % Ritorna nello spazio originale
        Xpred = ReverseEmbedding(Xnext_pred,embedding.window);
        
        % Calcola MSE fra predetto e pulito
        % MSE_values(i) = mean((Xpred-X(embedding.window*2:end,1)).^2,"all");
        
        % Riapplica embedding per D2
        Xnext_pred = Embedding(Xpred,embedding.variable,embedding.window);

        % Calcola D2
        D2_values(i) = CorrelationDimension_Krakosvka(Xnext_pred');

        disp("Iteration " + i);

    end
    
    hold on
    plot(noise_values,D2_values,'LineStyle','-','Color',colors(emb));
    % plot(noise_values,MSE_values,'LineStyle','--','Color',colors(emb));
    
    % yline(embedding.window,'--','Color',[0.5 0.5 0.5],'LabelHorizontalAlignment','left');
    
    drawnow
end

title('D2 dimension respect to the noise','FontSize',14);
xlabel('Noise level','FontSize',10);
ylabel('D2 dimension','FontSize',10);

set(gca,'XScale','log');
grid off;

legend({'embedding.window = 3','embedding.window = 5','embedding.window = 10','embedding.window = 20'},'Location','northwest');

savefig('Matteo_Scarcella/Figures/D2denoised-noise.fig')

%% D2 al variare della finestra di embedding (parametrizzata dal rumore)

n = 25;
embedding.variable = 1;
windows = linspace(1,n,n);
colors = ['r','g','b','c'];
noise_values = [1e-3,1e-2,1e-1,1e0];

figure(5)

for i = 1:length(noise_values)
   
    D2_values = zeros(n,1);

    for emb = 1:n
        embedding.window = emb;
        D2_values(emb) = RosslerD2(noise_values(i),embedding);
    end
    
    plot(windows,D2_values,'Color',colors(i),'LineWidth',1.5);
   
    % yline(embedding.window,'--','Color',[0.5 0.5 0.5],'LabelHorizontalAlignment','left');
    
    hold on
end

title('D2 dimension respect to the embedding size','FontSize',14);
xlabel('embedding size','FontSize',10);
ylabel('D2 dimension','FontSize',10);

grid off;

legend({'\sigma = 10^{-3}','\sigma = 10^{-2}','\sigma = 10^{-1}','\sigma = 10^0'},'Location','northwest');

savefig('Matteo_Scarcella/Figures/D2-embeddingsize.fig')

%% Loss respect to code size

% uso come sistema di test un Rossler (2) e uno spazio di punti di Sobol
% (5)

clear; close; clc;

embedding.window = 5;
embedding.variable = 1;
MaxCodeSize = 10;
Threshold = 0.1;
systems = ["Rossler","Sobol"];
colours = ["b","g"];

figure(6);
hold on
for i = 1:length(systems)

    if i == 1
        X = CoupledRossler(0);
        X = X(:,1);
        X = X';
        embedding.case = 1;
    elseif i == 2
        X = sobolset(5);
        X = X(1:10000,:);
        X = X';
        embedding.case = 0;
    end
        
    [D2,Loss] = AutoencoderCorrelationDimension(X,MaxCodeSize,Threshold,embedding);
    

    plot(Loss,'LineWidth',1.5,"Color",colours(i));
    plot(D2, Loss(D2), 'ro', 'MarkerSize', 10, 'LineWidth', 2,'HandleVisibility','off');

end

title('Loss respect code size','FontSize',14);
xlabel('Code size','FontSize',10);
ylabel('Loss (MSE)','FontSize',10);

grid off;

legend({"Rossler attractor","Sobol set"},'Location','northwest');

savefig('Matteo_Scarcella/Figures/Loss-codesize.fig')

%% D2 precision with different methods

% uso un test set, generando punti con distribuzione di Sobol, e calcolo la
% D2 al variare della dimensione target per valutare i diversi sistemi.

clear; close; clc;

n = [2,5,10,20]; % Come nell'articolo di Krakovska
samples = 10000;
methods = ["GrassbergerProcaccia","KrakovskaChvostekova","AutoencoderIDEA"];
colors = ['r','g','b'];
handler = '';

Structures = [64 32 16; 64 32 16; 128 64 32; 256 128 64];
CodeSizes = [8,8,16,32];

figure(7);
hold on
for method = 1:length(methods)

    D2_values = zeros(length(n),1);

    for target_dimension = 1:length(n)
        X = sobolset(n(target_dimension));
        X = X(1:samples,:);
        % X = normrnd(X,1);

        if method == 1
            D2_values(target_dimension) = correlationDimension(X);
        elseif method == 2
            D2_values(target_dimension) = CorrelationDimension_Krakosvka(X);
        elseif method == 3
            % [D2_values(target_dimension),~] = AutoencoderCorrelationDimension(X',ceil(n(target_dimension)*1.5),0.1);
            [D2_values(target_dimension),~,~,~] = IDEA(X',Structures(target_dimension,:),CodeSizes(target_dimension),0.001,1e2,5e3);
            disp("Iteration completed");
        end
    end

    plot(n,D2_values,'Color',colors(method),'LineWidth',1.5);
end

plot(n,n,'Color',[0.5,0.5,0.5],'LineStyle','--','LineWidth',1.5);

title('D2 dimension respect to the intrinsic dimension','FontSize',14);
xlabel('Intrinsic Dimension','FontSize',10);
ylabel('D2 dimension','FontSize',10);

grid off;

legend({'Grassberger&Procaccia','Krakovska&Chvostekova','AutoencoderIDEA','Desidered Behaviour'},'Location','northwest');

savefig('Matteo_Scarcella/Figures/D2-methods.fig')

%% D2 latent vs data
clear; close; clc;

[~,X] = LorenzAttractor(10,8/3,50);

embeddings = [3,5,10,20];

D2_data = zeros(length(embeddings),1);
D2_latent = zeros(length(embeddings),1);

for emb = 1:length(embeddings)

    embedding.window = embeddings(emb);
    embedding.variable = 1;
    [Xnext,~] = Embedding(X,embedding.variable,embedding.window);
    
    D2_data(emb) = CorrelationDimension_Krakosvka(Xnext');

    [~,~,LatentCode,~] = IDEA(Xnext,[128 64 32 16],8,0.001,100,5000);
    D2_latent(emb) = CorrelationDimension_Krakosvka(LatentCode');

    disp("Iteration completed");
end

figure(8);
hold on
plot(embeddings,D2_data,'-','LineWidth',2);
plot(embeddings,D2_latent,'-','LineWidth',2);

title('D2 krakovska from latent vs from data','FontSize',14);
xlabel('Embedding size','FontSize',10);
ylabel('D2 dimension','FontSize',10);

grid off;

legend({'D2 from original data','D2 from latent code'},'Location','northwest');

savefig('Matteo_Scarcella/Figures/D2idea-latent.fig')

%% D2 krakovska for different sample size of sobol set

clear; close; clc;

samples = logspace(2,4,20);

D2_krakovska = zeros(length(samples),1);
D2_procaccia = zeros(length(samples),1);

target_dimension = 3;

for sample = 1:length(samples)
    
    X = sobolset(target_dimension);
    X = X(1:samples(sample),:);

    Xnext = Embedding(X,[1 2 3],5);

    D2_krakovska(sample) = CorrelationDimension_Krakosvka(Xnext');
    D2_procaccia(sample) = correlationDimension(Xnext','dim',1);

end

figure(9);
hold on
plot(samples,target_dimension*ones(length(samples),1),'--','LineWidth',2,'Color',[0.5 0.5 0.5]);
plot(samples,D2_krakovska,'-','LineWidth',2);
plot(samples,D2_procaccia,'-','LineWidth',2);

title('D2 for different sample size of sobol set','FontSize',14);
xlabel('Sample size','FontSize',10);
ylabel('D2 dimension','FontSize',10);

set(gca,'XScale','log');

grid off;

legend({'Target','D2 krakovska','D2 procaccia'},'Location','northwest');

savefig('Matteo_Scarcella/Figures/D2-samplesize.fig')

%% D2 krakovska for different sample size of Lorenz attractor

clear; close; clc;

samples = [5,10,50,100,200,300];

D2_krakovska = zeros(length(samples),1);
D2_procaccia = zeros(length(samples),1);

target_dimension = 2;

for sample = 1:length(samples)
    
    [~,X] = LorenzAttractor(10,8/3,50,samples(sample));

    Xnext = Embedding(X,[1],5);
        
    D2_krakovska(sample) = CorrelationDimension_Krakosvka(Xnext');
    D2_procaccia(sample) = correlationDimension(Xnext','dim',1);

end

figure(9);
hold on
plot(samples,D2_krakovska,'-','LineWidth',2);
plot(samples,D2_procaccia,'-','LineWidth',2);
plot(samples,target_dimension*ones(length(samples),1),'--','LineWidth',2,'Color',[0.5 0.5 0.5]);
plot(samples,(target_dimension+1)*ones(length(samples),1),'--','LineWidth',2,'Color',[0.5 0.5 0.5]);


title('D2 for different sample size of lorenz attractor','FontSize',14);
xlabel('T end','FontSize',10);
ylabel('D2 dimension','FontSize',10);

set(gca,'XScale','log');

grid off;

legend({'D2 krakovska','D2 procaccia','Target'},'Location','northwest');

savefig('Matteo_Scarcella/Figures/D2-samplesize2.fig')

%% D2 krakovska from data and from latent for different sample size of Lorenz attractor

clear; close; clc;

samples = [10,50,100];

D2_data = zeros(length(samples),1);
D2_latent = zeros(length(samples),1);

target_dimension = 2;

for sample = 1:length(samples)
    
    [~,X] = LorenzAttractor(10,8/3,50,samples(sample));

    Xnext = Embedding(X,[1],5);
        
    D2_data(sample) = CorrelationDimension_Krakosvka(Xnext');

    [~,~,LatentCode,~] = IDEA(Xnext,[128 64 32 16],8,0.001,100,5000);
    D2_latent(sample) = CorrelationDimension_Krakosvka(LatentCode');

    disp("Iteration completed");
end

figure(9);
hold on
plot(samples,D2_data,'-','LineWidth',2);
plot(samples,D2_latent,'-','LineWidth',2);
plot(samples,target_dimension*ones(length(samples),1),'--','LineWidth',2,'Color',[0.5 0.5 0.5]);
plot(samples,(target_dimension+1)*ones(length(samples),1),'--','LineWidth',2,'Color',[0.5 0.5 0.5]);


title('D2 from data and from latent code for different sample size of lorenz attractor','FontSize',14);
xlabel('T end','FontSize',10);
ylabel('D2 dimension','FontSize',10);

set(gca,'XScale','log');

grid off;

legend({'D2 data','D2 latent code','Target'},'Location','northwest');

savefig('Matteo_Scarcella/Figures/D2-samplesize3.fig')

%% D2 con autoencoder IDEA all'aumentare dell'embedding size per rumore e lorenz
clear; close; clc;

embeddings = [1,3,5,10,20];

IDlorenz = zeros(length(embeddings),1);
IDnoise = zeros(length(embeddings),1);

for emb = 1:length(embeddings)

    Xlorenz = LorenzAttractor(10,8/3,50,50);
    Xlorenz = Embedding(Xlorenz,1,embeddings(emb));

    Xnoise = randn(10000,1);
    Xnoise = Embedding(Xnoise,1,embeddings(emb));

    [IDlorenz(emb),~,~,~] = IDEA(Xlorenz,[256 128 64 32 16],8,0.001,100,5000);
    [IDnoise(emb),~,~,~] = IDEA(Xnoise,[256 128 64 32 16],8,0.001,100,5000);
    
    disp("Iteration completed");
end

figure(10);
hold on
plot(embeddings,IDlorenz,'-','LineWidth',2);
plot(embeddings,IDnoise,'-','LineWidth',2);

title('ID from data and from latent code for different sample size of lorenz attractor','FontSize',14);
xlabel('Embedding size','FontSize',10);
ylabel('D2 dimension','FontSize',10);

grid off;

legend({'ID Lorenz','ID noise'},'Location','northwest');

savefig('Matteo_Scarcella/Figures/D2idea-lorenzvsnoise.fig')

%% D2 latent vs data (normal auto-encoder)
clear; close; clc;

[~,X] = LorenzAttractor(10,8/3,50);

embeddings = [3,5,10,20];

D2_data = zeros(length(embeddings),1);
D2_latent = zeros(length(embeddings),1);

for emb = 1:length(embeddings)

    embedding.window = embeddings(emb);
    embedding.variable = 1;
    [Xnext,~] = Embedding(X,embedding.variable,embedding.window);
    
    D2_data(emb) = CorrelationDimension_Krakosvka(Xnext');
    
    [~,LatentCode] = Autoencoder(Xnext,[64 32 16],3);
    % [~,~,LatentCode,~] = IDEA(Xnext,[128 64 32 16],8,0.001,100,5000);
    D2_latent(emb) = CorrelationDimension_Krakosvka(LatentCode');

    disp("Iteration completed");
end

figure(8);
hold on
plot(embeddings,D2_data,'-','LineWidth',2);
plot(embeddings,D2_latent,'-','LineWidth',2);

title('D2 krakovska from latent vs from data','FontSize',14);
xlabel('Embedding size','FontSize',10);
ylabel('D2 dimension','FontSize',10);

grid off;

legend({'D2 from original data','D2 from latent code'},'Location','northwest');

savefig('Matteo_Scarcella/Figures/D2-latent.fig')

