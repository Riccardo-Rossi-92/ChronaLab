clear; clc;

% Example 02 - Lorenz embedding analysis

t_end = 100;
N = 10000;
X0 = [1 1 1];
method = "ODE45";

[X,t]  = LorenzAttractor(t_end,N,X0,method);

%% Embedding analysis

% variable: 
x = X(:,1);

N = 3:10;
W = 1:5;

D2 = zeros(length(W),length(N));

for i = 1 : length(N)
    for j = 1 : length(W)

        n = N(i);
        w = W(j);

        if n > w
            X_buf = buffer(x,n,n-w);
            x_buf = X_buf(:,n:end);
            x_buf = x_buf';
            x_buf = x_buf./std(x);
            D2(j,i) = CorrelationDimension(X_buf,"MATLAB");
        end

    end
end

figure(1)
clf
contourf(N,W,D2,30,'LineStyle','none')
colorbar;
xlabel('Embedding Dimension N');
ylabel('Window Size W');
title('Correlation Dimension Analysis');


