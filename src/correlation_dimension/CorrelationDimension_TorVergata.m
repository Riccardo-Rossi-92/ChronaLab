function [D2,D2std] = CorrelationDimension_TorVergata(X,RegressionType,K)

if nargin == 1
    RegressionType = "Removal"; % "Standard", "Weighted; "Moving" ; "Plateau"
    K = 200;
elseif nargin == 2
    K = 200;
end

RegressionWindow = 30;

N = size(X,1);

% find K-nearest
[~, D] = knnsearch(X, X, 'K', K+1);
D = D(:,2:end);

% average distance
r = mean(D, 1);

% correlation
C = (1:K) / N;
% C = r / N;

% log-scale
log_r = log(r);
log_C = log(C);

if RegressionType == "Standard"

    M = [log_r; ones(size(log_r))];
    % b = linsolve(M',log_C');
    b = M' \ log_C';
        
    D2 = b(1);
    D2std = nan;
elseif RegressionType == "Weighted"


    M = [log_r; ones(size(log_r))];
    w = C.^2;
    b = linsolve((w.*M)',(w.*log_C)');

    D2 = b(1);
    D2std = nan;
elseif RegressionType == "Moving"

    n = length(r);
    best_R2 = 0;

    for i = 1:(n-RegressionWindow+1)

        idx = i:(i+RegressionWindow-1);
        x = log_r(idx);
        y = log_C(idx);
        [p, S] = polyfit(x,y,1);
        ynow = polyval(p,x);

        SSres = sum((y - ynow).^2);
        SStot = sum((y - mean(y)).^2);
        R2 = 1 - SSres/SStot;

        if R2 > best_R2
            best_R2 = R2;
            D2 = p(1);
            idx_best = idx;
        end
    end
    D2std = nan;

elseif RegressionType == "Removal"

   convergence = true;

   x = log_r;
   y = log_C;

   best_R2 = 0;

   while convergence

       p = polyfit(x,y,1);
       y_p = polyval(p,x);
      
       SSres = sum((y - y_p).^2);
       SStot = sum((y - mean(y)).^2);
       R2 = 1 - SSres/SStot.*(length(x)-1)./(length(x)-3);

       if R2 > best_R2
            best_R2 = R2;
            D2 = p(1);
       else 
           convergence = false;
       end

       [~,ind_max] = max(abs(y_p-y));

       x(ind_max) = [];
       y(ind_max) = [];

       if length(x) < 10
           convergence = false;
       end

       D2std = nan;
   end



elseif RegressionType == "Plateau"

    slope_local = diff(log_C)./diff(log_r);

    k = 5;
    slope_smooth = movmean(slope_local,k);
    slope_smooth = rmoutliers(slope_smooth);

    D2 = mean(slope_smooth);
    D2std = std(slope_smooth);


end

end