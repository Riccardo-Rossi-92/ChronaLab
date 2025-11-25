function Emb = Embedding_System(X, embedding)

for j = 1 : embedding.shifts

    E = [];

    for i = 1 : length(embedding.variable)

        x = X(:,embedding.variable(i));
        Xt = buffer(x,embedding.window,embedding.window-embedding.resolution);

        Xt = Xt(:,embedding.window:end);

        E = [E; Xt];

    end

    N = size(E,2);

    Emb{j}.E = E(:,(embedding.window + embedding.steps*(j-1)):(N-embedding.steps*(embedding.shifts-j)));
    Emb{j}.ind = (embedding.window + embedding.steps*(j-1)):(N-embedding.steps*(embedding.shifts-j));
    
end