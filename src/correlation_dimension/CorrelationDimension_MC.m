

function [D2_mean, D2_std, Loss] = CorrelationDimension_MC(X,MC,type,perc)

    %% Input control 
    
    if nargin == 0
        disp("Not enough input")
        return;
    elseif nargin == 1
        MC = 30; 
        type = "Krakosvka";
        perc = 0.1;
    elseif nargin == 2
        type = "Krakosvka";
        perc = 0.1;
    elseif nargin == 3
        perc = 0.1;
    end

    %% Monte Carlo evaluation

    N = size(X,1);
    ind = 1 : N;
    Loss = 0;

    if type == "Procaccia"
        for mc = 1 : MC
            ind_rand = randsample(ind,floor((1-perc)*N));
            D2_values(mc) = correlationDimension(X(ind_rand,:),0,1);
        end
        
    elseif type == "Krakosvka"
        for mc = 1 : MC
            ind_rand = randsample(ind,floor((1-perc)*N));
            D2_values(mc) = CorrelationDimension_Krakosvka(X(ind_rand,:));
        end
    
    elseif type == "TorVergata"
        for mc = 1 : MC
            ind_rand = randsample(ind,floor((1-perc)*N));
            D2_values(mc) = CorrelationDimension_TorVergata(X(ind_rand,:));
        end
        
    elseif type == "IDEA-2"
        % D2_values = IDEA2(X',[128 64 32 16],8,0.005,0);
        D2_values = IDEA2(X',[128 64 32 16],8,[0.001,0.01]);


    elseif type == "IDEA"
        for mc = 1:5
            D2_values(mc) = IDEA(X',[128 64 32 16],6);
        end

    elseif type == "TwoNN"
        D2_values = TwoNN(X);
    end
    

    D2_mean = mean(D2_values);
    D2_std = std(D2_values);

end