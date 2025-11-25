function [X,Y,t,info] = CoupledRossler(C,t_end,N,X0,Y0,method)

%% consider various cases

if nargin == 5
    method = "ODE45";
elseif nargin == 4
    method = "ODE45";
    Y0 = [0,0,0.4];
elseif nargin == 3
    method = "ODE45";
    Y0 = [0,0,0.4];
    X0 = [0,0,0.4];
elseif nargin == 2
    method = "ODE45";
    Y0 = [0,0,0.4];
    X0 = [0,0,0.4];
    N = 1e4;
elseif nargin == 1
    method = "ODE45";
    Y0 = [0,0,0.4];
    X0 = [0,0,0.4];
    N = 1e4;
    t_end = N*0.1;
elseif nargin == 0
    method = "ODE45";
    Y0 = [0,0,0.4];
    X0 = [0,0,0.4];
    N = 1e4;
    t_end = N*0.1;
    C = 0;
end

%% first 1000 are not considered

N = N + 1000;

%% Time
t = linspace(0, t_end, N); % Create a time vector from 0 to t_end with N points

%% Initialize the state variables
X = zeros(N, 3); % Preallocate the solution array for the Lorenz system
X(1, :) = X0; % Initial conditions

Y = zeros(N, 3); % Preallocate the solution array for the Lorenz system
Y(1, :) = Y0; % Initial conditions

%% Simulate

if method == "EulerBackward"

    dt = mean(diff(t));

    for i = 2 : N

        X(i,1) = X(i-1,1) + (-1.015*X(i-1,2)-X(i-1,3))*dt;
        X(i,2) = X(i-1,2) + (1.015*X(i-1,1)+0.15*X(i-1,2))*dt;
        X(i,3) = X(i-1,3) + (0.2 + X(i-1,3)*(X(i-1,1)-10))*dt;

        Y(i,1) = Y(i-1,1) + (-0.985*Y(i-1,2)-Y(i-1,3)+C*(X(i-1,1)-Y(i-1,1)))*dt;
        Y(i,2) = Y(i-1,2) + (0.985*Y(i-1,1)+0.15*Y(i-1,2))*dt;
        Y(i,3) = Y(i-1,3) + (0.2 + Y(i-1,3)*(Y(i-1,1)-10))*dt;

    end
end

if method == "ODE45"

     % Runge-Kutta 4th/5th order ODE solver

    f = @(t,a) [-1.015*a(2)-a(3);...
                1.015*a(1)+0.15*a(2);...
                0.2 + a(3)*(a(1)-10);...
                -0.985*a(5) - a(6) + C*(a(1)-a(4));...
                0.985*a(4) + 0.15*a(5);...
                0.2 + a(6)*(a(4)-10)];
        
   [t_ode,Z] = ode45(f,[0 t_end],[X0  Y0]);    

   Z = interp1(t_ode,Z,t);

end

X = Z(1000:end,1:3);
Y = Z(1000:end,4:6);
t = t(1000:end);


end