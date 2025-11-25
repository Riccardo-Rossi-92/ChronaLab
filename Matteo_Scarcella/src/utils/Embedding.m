function [Xnext,Xprev] = Embedding(X, variable, window)

    Xnext = [];
    Xprev = [];

    for i = 1 : length(variable)

        % prende la colonna i-esima, cioè la i-esima coordinata (10001x1)
        x = X(:,variable(i)); 
        
        % trasforma un vettore colonna in una matrice con window
        % righe e lenght(x) colonne.
        % ogni colonna è una finestra temporale che scorre in avanti dal
        % basso verso l'alto
        Xt = buffer(x,window,window-1);

        % scarta le prime 4 colonne, che contengono degli 0 di padding
        Xt = Xt(:,window:end);

        % prende gli ultimi 9992 valori
        Xnext = [Xnext; Xt(:,window+1:end)];

        % prende i primi 9992 valori
        Xprev = [Xprev; Xt(:,1:end-window)];
    end
    clear x Xt

end