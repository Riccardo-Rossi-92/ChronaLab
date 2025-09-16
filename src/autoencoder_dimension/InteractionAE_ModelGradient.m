
function [gradient,Loss] = InteractionAE_ModelGradient(parameters,AE,dlXn,dlXp,dlYn,dlYp,dlXS,dlYS)
    
    [dlXpred,dlYpred] = InteractionAE(dlXp,dlYp,1,parameters,AE);
    
    % MSE
    MSE = mean(mean((dlXpred-dlXn).^2./dlXS.^2,2),1) + ...
            mean(mean((dlYpred-dlYn).^2./dlYS.^2,2),1);

    Loss = MSE;

    % gradient
    gradient = dlgradient(Loss,parameters);

end



