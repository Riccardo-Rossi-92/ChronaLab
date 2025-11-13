
function [gradient,MSE,CE,Loss] = VAE_ModelGradient_Pred(parameters,AE,dlXn,dlXp,dlS)
    
    % Predict
    [dlXpp,dlCp,p] = VAE_Network_Hybrid(dlXp,1,parameters,AE); 
    
    % MSE
    MSE = mean(mean((dlXpp-dlXp).^2./dlS.^2,2),1);

    % code reg
    Code_mean = mean(dlCp,2);
    Reg = mean(Code_mean.^2);

    %% Penalty for soft code
    CE = -mean(1*(log(max(1-p,eps))));
    

    % total loss
    Loss = MSE + Reg + AE.alpha.*CE;

    % gradient
    gradient = dlgradient(Loss,parameters);

end



