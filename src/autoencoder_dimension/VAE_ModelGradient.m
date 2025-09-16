
function [gradient,MSE,Reg,Loss] = VAE_ModelGradient(parameters,AE,dlXn,dlXp,dlS)
    
    [dlXpred,dlCode] = VAE_Network(dlXp,1,parameters,AE);
    
    % MSE
    MSE = mean(mean((dlXpred-dlXn).^2./dlS.^2,2),1);

    % code reg
    mean_code = mean(dlCode,2);
    Reg = mean(mean_code.^2);

    % total loss
    Loss = MSE + Reg;

    % gradient
    gradient = dlgradient(Loss,parameters);

end



