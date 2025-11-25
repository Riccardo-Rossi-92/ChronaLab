function [X,Y] = CoupledRossler2(C,t_end,N)

if nargin == 1
    t_end = 100;
    N = t_end*100;
elseif nargin == 2
    N = t_end*100;
end

% Time
tspan = linspace(0, t_end, N);

% Initial conditions
X0 = [0;0;0.4];
Y0 = [0;0;0.4];

% ode45 → Runge–Kutta adattivo di ordine medio (generico)
% ode15s → per sistemi stiff
% ode23s → più veloce ma meno preciso
options = odeset('RelTol',1e-6,'AbsTol',1e-8);

[t,sol] = ode45(@(t,x) system(t,x,C), tspan, [X0;Y0], options);

% rimuovo i primi secondi di assestamento
start = 15000;

X = sol(start:end,1:3);
Y = sol(start:end,4:6);

function dx = system(t, x, C)
    
    dx = zeros(6,1);
    
    % Equazioni sistema di Rössler
    dx(1) = -x(2) - x(3);
    dx(2) = x(1) + 0.1*x(2);
    dx(3) = 0.1 + x(3)*(x(1)-14);

    dx(4) = -0.985*x(5) - x(6) + C*(x(1)-x(4));
    dx(5) = 0.985*x(4) + 0.15*x(5);
    dx(6) = 0.2 + x(6)*(x(4)-10);
end

end