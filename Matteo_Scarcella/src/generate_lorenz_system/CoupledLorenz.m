function [t,x] = CoupledLorenz(C,tend,N)

sigma = 10;
beta = 8/3;
rho = 28;

tspan = linspace(0,tend,tend*100);         
% x0 = [1; 0; 0; 1.1; 0; 0];
x0 = [1; 0; 0; 1; 0; 0];

% ode45 → Runge–Kutta adattivo di ordine medio (generico)
% ode15s → per sistemi stiff
% ode23s → più veloce ma meno preciso
options = odeset('RelTol',1e-6,'AbsTol',1e-8);

[t,x] = ode45(@(t,x) lorenz(t,x), tspan, x0, options);

start = 2500;
x = x(start:end,:);

    function dx = lorenz(t, x)
        
        dx = zeros(6,1);
        
        % Equazioni sistema di Lorenz
        % dx(1) = sigma*(x(2)-x(1)) + C*((x(4)-x(1)) + (x(5)-x(2)) + (x(6)-x(3)));
        % dx(2) = x(1)*(rho-x(3)) - x(2); 
        % dx(3) = x(1)*x(2) - beta*x(3);

        dx(1) = sigma*(x(2)-x(1)) + C*(x(4)-x(1));
        dx(2) = x(1)*(rho-x(3)) - x(2); 
        dx(3) = x(1)*x(2) - beta*x(3);

        dx(4) = sigma*(x(5)-x(4));
        dx(5) = x(4)*((rho+2)-x(6)) - x(5);
        dx(6) = x(4)*x(5) - beta*x(6);
    end

end