function [D2,Loss] = AutoencoderCorrelationDimension(X,MaxCodeSize,ElbowThreshold,embedding)
% Accetta una matrice di dimensione MxN, dove M è il numero dei canali e N
% è il numero dei campioni

if nargin == 3
    embedding.case = 0;
end

Loss = zeros(MaxCodeSize,1);

Xdefault = X;

% Finding the loss values
for CodeSize = MaxCodeSize:-1:1
    
    X = Xdefault;

    % Embedding
    if (embedding.case == 1)
        [X,~] = Embedding(X',embedding.variable,embedding.window);
    end

    Xpred = Autoencoder(X,[64 32 16],CodeSize);

    % De-Embedding
    if (embedding.case == 1)
        X = ReverseEmbedding(X,embedding.window);
        Xpred = ReverseEmbedding(Xpred,embedding.window);
    end

    Loss(CodeSize) = mean((Xpred-X).^2,"all");

    disp("Iteration completed - Code size: " + CodeSize);

end

% Find the elbow of the loss (corresponding to the D2 value)
D2 = 1;
for i = 1:length(Loss)

    if (Loss(i)-min(Loss)) / (max(Loss)-min(Loss)) <= ElbowThreshold
        D2 = i;
        break;
    end

end

end