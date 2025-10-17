function X = ReverseEmbedding(Xembedded,embedding_window)
    
    % Taken from ReverseBuffer script
    n = size(Xembedded,2);
    
    i = 1 : n;
    i = buffer(i,embedding_window,embedding_window-1);
    
    X = zeros(1,n);
    
    for j = 1 : n
    
        indices = i==j;
    
        X(j) = mean(Xembedded(indices));
    
    end

    X = X';

end

