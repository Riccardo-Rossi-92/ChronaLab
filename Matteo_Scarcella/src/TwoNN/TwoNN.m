function ID = TwoNN(X)
% https://github.com/jmmanley/two-nn-dimensionality-estimator/blob/master/twonn.py

    N = size(X,1);
    K = 3;

    [~, D] = knnsearch(X, X, 'K', K);

    for i = 1:N

        mu(i) = D(i,3) / D(i,2);

    end

    [~, sort_idx] = sort(mu);

    % Femp = np.arange(N)/N
    Femp = (0:(N-1))' / N;

    % x = np.log(mu[sort_idx]).reshape(-1,1)
    % y = -np.log(1-Femp).reshape(-1,1)
    x = log(mu(sort_idx));
    y = -log(1 - Femp);

    % FIT straight line through origin:
    % slope = (x^T y) / (x^T x)
    ID = dot(x,y) / dot(x,x);

end
