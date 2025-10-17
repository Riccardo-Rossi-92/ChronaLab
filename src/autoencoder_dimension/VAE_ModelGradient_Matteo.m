
function [gradient,MSE,Reg,Loss] = VAE_ModelGradient_Matteo(parameters,AE,dlXn)
    
    [dlXpred,dlCode] = VAE_Network_Matteo(dlXn,1,parameters,AE);
    
    % MSE
    MSE = mean((dlXpred - dlXn).^2, 'all');

    % L2 Regularization
    mean_code = mean(dlCode,2);
    Reg = mean(mean_code.^2);

    % total loss
    Loss = MSE;

    % gradient
    gradient = dlgradient(Loss,parameters);

end



