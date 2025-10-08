
function [gradient,MSE,Reg,Loss] = VAE_ModelGradient_Matteo(parameters,AE,dlXnoise,dlXtarget)
    
    [dlXpred,dlCode] = VAE_Network_Matteo(dlXnoise,1,parameters,AE);
    
    % MSE
    % dlXnoise2 = dlXtarget + randn(size(dlXtarget));
    MSE = mean((dlXpred - dlXtarget).^2, 'all');

    % L2 Regularization
    mean_code = mean(dlCode,2);
    Reg = mean(mean_code.^2);

    % total loss
    Loss = MSE;

    % gradient
    gradient = dlgradient(Loss,parameters);

end



