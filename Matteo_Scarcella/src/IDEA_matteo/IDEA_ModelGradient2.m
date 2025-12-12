
function [gradient,Loss] = IDEA_ModelGradient2(parameters,AE,dlXn)
    
    [dlXpred,~] = IDEA_Network2(dlXn,1,parameters,AE);
    
    % MSE
    MSE = mean((dlXpred - dlXn).^2, 'all');
    Norm = mean((dlXn).^2,'all');
    NMSE = MSE / Norm;

    % Total loss
    Loss = NMSE;

    

    % gradient
    gradient = dlgradient(Loss,parameters);

end



