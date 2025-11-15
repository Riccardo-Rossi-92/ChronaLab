function X = ReverseEmbedding(Xembedded,embedding_window,variable)

    if nargin == 2
        variable = 1;
    end
    
    % Taken from ReverseBuffer script
    n = size(Xembedded,2);
    
    i = 1 : n;
    buff = zeros(embedding_window*variable,n);
    for var = 1 : variable
        buff(((var-1)*embedding_window)+1:(var*embedding_window),:) = buffer(i,embedding_window,embedding_window-1);
    end
    
    X = zeros(variable,n);
    
    for j = 1 : n
        
        
        for var = 1 : variable
            mask = false(embedding_window*variable,n);
            mask(((var-1)*embedding_window)+1:(var*embedding_window),:) = buff(((var-1)*embedding_window)+1:(var*embedding_window),:)==j;

            X(var,j) = mean(Xembedded(mask));
        
        end
    end

    X = X';

end

