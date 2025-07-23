function [X,t,info] = LorenzAttractor(t_end,N,X0,method,param)

%% consider various cases

if nargin == 4
    param = [10; 28; 8/3];
elseif nargin == 3
    method = "EulerBackward";
    param = [10; 28; 8/3];
elseif nargin == 2
    X0 = [1,1,1];
    method = "EulerBackward";
    param = [10; 28; 8/3];
elseif nargin == 1
    N = 1000;
    X0 = [1,1,1];
    method = "EulerBackward";
    param = [10; 28; 8/3];
elseif nargin == 0
    t_end = 10;
    N = 1000;
    X0 = [1,1,1];
    method = "EulerBackward";
    param = [10; 28; 8/3];
end

%% Time
t = linspace(0, t_end, N); % Create a time vector from 0 to t_end with N points

%% Initialize the state variables
X = zeros(N, 3); % Preallocate the solution array for the Lorenz system
X(1, :) = X0; % Initial conditions

%% Lorenz parameters

sigma = param(1);
rho = param(2);
beta = param(3);

%% Simulate

if method == "EulerBackward"

    dt = mean(diff(t));

    for i = 2 : N

        X(i,1) = X(i-1,1) + sigma*(X(i-1,2)-X(i-1,1)).*dt;
        X(i,2) = X(i-1,2) + (X(i-1,1).*(rho - X(i-1,3)) - X(i-1,2)).*dt;
        X(i,3) = X(i-1,3) + (X(i-1,1).*X(i-1,2)-beta.*X(i-1,3)).*dt;

    end
end

if method == "ODE45"

     % Runge-Kutta 4th/5th order ODE solver

    f = @(t,a) [-sigma*a(1) + sigma*a(2); rho*a(1) - a(2) - a(1)*a(3); -beta*a(3) + a(1)*a(2)];
    [t_ode,X] = ode45(f,[0 t_end],X0);    

    X = interp1(t_ode,X,t);

end

end