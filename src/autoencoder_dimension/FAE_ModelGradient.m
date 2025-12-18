
function [gradient,MSE,Reg,Loss] = FAE_ModelGradient(parameters,AE,dlXn,dlXp,dlS)
    
    [dlXpred,dlCode,dlCodeS] = FAE_Network(dlXp,1,parameters,AE);
    
    % MSE
    MSE = mean(mean((dlXpred-dlXn).^2./dlS.^2,2),1);

    % code reg
    mean_code = mean(dlCode,2);
    std_code = std(dlCode,[],2);
    std_codeS =std(dlCodeS,[],2);
    
    
    Reg = sum(mean_code.^2) + sum(std_code) + sum(std_codeS);

    % total loss
    Loss = MSE + Reg;

    % gradient
    gradient = dlgradient(Loss,parameters);

end



