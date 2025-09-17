clear; clc;

%% Generate system

C = 0.0;
[X,Y] = CoupledRossler(C);
X = [X Y];

% add noise

%% time-series based configuration

embedding.case = 1;
embedding.variable = [1 4];
embedding.window = 5;

%% Embedding

if embedding.case == 1
    Xnext = [];
    Xprev = [];

    for i = 1 : length(embedding.variable)

        x = X(:,embedding.variable(i));
        Xt = buffer(x,embedding.window,embedding.window-1);

        Xt = Xt(:,embedding.window:end);

        Xnext = [Xnext; Xt(:,embedding.window+1:end)];
        Xprev = [Xprev; Xt(:,1:end-embedding.window)];
    end
    clear Xt

elseif embedding.case == 0

    X = X';

    Xnext = X(:,2:end);
    Xprev = X(:,1:end-1);

end


%%

D2 = CorrelationDimension_Krakosvka(Xnext');






