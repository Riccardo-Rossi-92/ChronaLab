
function [gradient,MSE,Reg,Loss] = AE_ModelGradient(parameters,AE,dlX,dlS)
    
    [dlXp,dlC] = VAE_Network(dlX,1,parameters,AE);
    
    % MSE
    MSE = mean(mean((dlXp-dlX).^2./dlS.^2,2),1);

    % code reg
    mean_code = mean(dlC,2);
    Reg = mean(mean_code.^2);

    % total loss
    Loss = MSE + Reg;

    % gradient
    gradient = dlgradient(Loss,parameters);

end



