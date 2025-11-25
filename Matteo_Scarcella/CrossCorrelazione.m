%% Calcola la cross-correlazione fra variabili di sistemi accoppiati
clear; close; clc;

space = 1:0.05:3.5;
R = zeros(length(space), 2001);

for i = 1:length(space)
    
    [~,X] = CoupledLorenz(space(i),200,20000);
    
    % calcola cross-correlazione
    [r, lags] = xcorr(X(:,1), X(:,4), 1000, 'coeff');
    R(i, :) = r;
    
end

figure(1);
surf(space, lags, R');
shading interp
xlabel('C');
ylabel('Lag');
zlabel('Cross-Correlazione');
colorbar