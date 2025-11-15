function [t,x] = LorenzAttractor(sigma,beta,rho,tend)

if nargin == 3
    tend = 100;
end

tspan = linspace(0,tend,tend*100);         
x0 = [0.1; 0; 0];           

% ode45 → Runge–Kutta adattivo di ordine medio (generico)
% ode15s → per sistemi stiff
% ode23s → più veloce ma meno preciso
options = odeset('RelTol',1e-6,'AbsTol',1e-8);

[t,x] = ode45(@(t,x) lorenz(t,x), tspan, x0, options);

    function dx = lorenz(t, x)
        
        dx = zeros(3,1);
        
        % Equazioni sistema di Lorentz
        dx(1) = sigma*(x(2)-x(1));
        dx(2) = x(1)*(rho-x(3)) - x(2);
        dx(3) = x(1)*x(2) - beta*x(3);
    end

end