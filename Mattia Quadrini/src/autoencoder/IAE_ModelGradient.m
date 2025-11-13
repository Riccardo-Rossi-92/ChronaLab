
function [gradient,Loss] = IAE_ModelGradient(parameters,AE,dlXn,dlXp,dlYn,dlYp,dlXS,dlYS)
    
    % Predict
    [dlXpred,dlYpred] = IAE_Network(dlXp,dlYp,1,parameters,AE);
    
    % MSE
    MSE = mean(mean((dlXpred-dlXn).^2./dlXS.^2,2),1) + ...
            mean(mean((dlYpred-dlYn).^2./dlYS.^2,2),1);

    % Total Loss
    Loss = MSE;

    % gradient
    gradient = dlgradient(Loss,parameters);

end



