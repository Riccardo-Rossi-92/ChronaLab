
function [gradient,Loss,MSE,Lrec,Reg,Lort] = IDEA_ModelGradient(parameters,AE,dlXn)
    
    [dlXpred,Latent] = IDEA_Network(dlXn,1,parameters,AE);
    [dlXproj,~] = IDEA_Network(dlXn,1,parameters,AE,1);
    
    % MSE
    MSE = mean((dlXpred - dlXn).^2, 'all');

    % Projected Reconstruction
    Lrec = mean((dlXproj - dlXn).^2, 'all');

    % L1 Reg
    idx = size(Latent,1) - length(parameters.CO1.weights(parameters.CO1.weights>0)) + 1;
    alpha = 0.001;
    Reg = abs(parameters.CO1.weights(idx) + alpha);

    % Orthogonality
    covmatrix = cov(reshape(Latent,[size(Latent,2),size(Latent,1)]));
    l = size(Latent,1);
    idxx = 1:l+1:l*l;
    d = covmatrix(idxx);
    devmatrix = d' * d;
    corrmatrix = covmatrix ./ devmatrix;
    Lort = sum(corrmatrix.^2, 'all');

    % total loss
    lambda_rec = 1;
    lambda_reg = 1;
    lambda_ort = 0.1;
    Loss = 1*MSE + lambda_rec * Lrec + lambda_reg * Reg + lambda_ort * Lort;

    % gradient
    gradient = dlgradient(Loss,parameters);

end



