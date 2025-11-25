function [ID,Loss] = AutoencoderCorrelationDimension(X,MaxCodeSize,ElbowThreshold,embedding)
% Accetta una matrice di dimensione NxM, dove M è il numero dei canali e N
% è il numero dei campioni

X = X';

if nargin == 3
    embedding.case = 0;
end

Loss = zeros(MaxCodeSize,1);

% Finding the loss values
for CodeSize = MaxCodeSize:-1:1
    
    [~,~,CurrentLoss] = Autoencoder(X,[16 8],CodeSize);

    Loss(CodeSize) = CurrentLoss;

    disp("Iteration completed - Code size: " + CodeSize);

end

% Calcola il punto a gomito
LogLossDerivativeNorm = rescale(diff(movmean(log(Loss),1)).^2,0,1); % Il quadrato penalizza le grosse variazioni rispetto a quelle piccole (???)

k = 0;
ID = 0;
for i = 1:MaxCodeSize-1
    if LogLossDerivativeNorm(i) < ElbowThreshold % Soglia da scegliere
        k = k + 1;
        if ID == 0 % Se non era ancora stata assegnata
            ID = i;
        end
    else
        k = 0;
        ID = 0;
    end

    % Se per almeno 4 volte ottengo una pendenza sotto alla soglia...
    if k >= 4
        break;
    end
end

end