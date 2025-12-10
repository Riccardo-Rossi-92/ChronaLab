%% Calcola la cross-correlazione fra variabili di sistemi accoppiati
clear; close; clc;

space = 1:0.05:10;
R = zeros(length(space), 2001);

for i = 1:length(space)
    
    [~,X] = CoupledLorenz(space(i),100,10000);
    
    % calcola cross-correlazione
    [r, lags] = xcorr(X(:,3), X(:,6), 1000, 'coeff');
    R(i, :) = r;
    
end

figure(1);
surf(space, lags, R');
shading interp
xlabel('C');
ylabel('Lag');
zlabel('Cross-Correlazione');
colorbar


saveas(gcf, '/Users/matteoscarcella/Documents/Università/Tesi/Grafici/CrossCorrelazione3.svg', 'svg');   % salva in SVG