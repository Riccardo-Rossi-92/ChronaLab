clear; clc; close;

%% D2 al variare del rumore

n = 100;
embedding.variable = 1;
windows = [3,5,10];
colors = ['r','g','b'];

for emb = 1:length(windows)
    
    embedding.window = windows(emb);

    noise_values = linspace(1e-4,1e0,n);
    D2_values = zeros(n,1);
    
    for i = 1:n
        D2_values(i) = RosslerD2(noise_values(i),embedding);
    end
    
    figure(1);
    plot(noise_values,D2_values,'Color',colors(emb));
    
    title('D2 dimension respect to the noise','FontSize',14);
    xlabel('Noise level','FontSize',10);
    ylabel('D2 dimension','FontSize',10);
    
    set(gca,'XScale','log');
    grid off;
    
    % yline(embedding.window,'--','Color',[0.5 0.5 0.5],'LabelHorizontalAlignment','left');
    
    hold on
end

legend({'embedding.window = 3','embedding.window = 5','embedding.window = 10'},'Location','northwest');

%% D2 al variare della finestra di embedding (parametrizzata dal rumore)

n = 20;
embedding.variable = 1;
windows = linspace(1,n,n);
colors = ['r','g','b','c'];
noise_values = [1e-3,1e-2,1e-1,1e0];

for i = 1:length(noise_values)
   
    D2_values = zeros(n,1);

    for emb = 1:n
        embedding.window = emb;
        D2_values(emb) = RosslerD2(noise_values(i),embedding);
    end
    
    figure(2);
    plot(windows,D2_values,'Color',colors(i));
    
    title('D2 dimension respect to the embedding size','FontSize',14);
    xlabel('embedding size','FontSize',10);
    ylabel('D2 dimension','FontSize',10);
    
    % set(gca,'XScale','log');
    grid off;
    
    % yline(embedding.window,'--','Color',[0.5 0.5 0.5],'LabelHorizontalAlignment','left');
    
    hold on
end

legend({'noise = 1e-3','noise = 1e-2','noise = 1e-1','noise = 1e0'},'Location','northwest');
