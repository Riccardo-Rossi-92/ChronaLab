function D2 = RosslerD2(noise_level, embedding)

    % Crea un sistema di Rossler
    [X,~] = CoupledRossler(0);

    % Aggiunge rumore
    if noise_level ~= 0
        X = normrnd(X,noise_level);
    end
    
    % Applica embedding
    [Xnext,~] = Embedding(X,embedding.variable,embedding.window);

    % Calcola D2-dimension
    D2 = CorrelationDimension_Krakosvka(Xnext');

end