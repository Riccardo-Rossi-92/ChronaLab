clear; clc; close;

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
        [D2_mean(i), D2_std(i)] = CorrelationDimension_MC(Xnext');

        disp("Iteration " + i);
    end

    hold on
    errorbar(noise_values,D2_mean,D2_std,'Color',colors(emb));
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