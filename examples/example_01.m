clear; clc;

% Example 01 - Lorenz genration and test

t_end = 30;
N = 20000;
X0 = [1 1 1]
method = "EulerBackward";

[X_Euler,t]  = LorenzAttractor(t_end,N,X0,method);

method = "ODE45";
[X_ODE45,t]  = LorenzAttractor(t_end,N,X0,method);


figure(1)
clf
subplot(3,2,1)
plot(t,X_Euler(:,1),'-b')
hold on
plot(t,X_ODE45(:,1),'-r')
grid on
grid minor
xlabel("t")
ylabel("x")
legend("Euler","ODE45")

subplot(3,2,3)
plot(t,X_Euler(:,2),'-b')
hold on
plot(t,X_ODE45(:,2),'-r')
grid on
grid minor
xlabel("t")
ylabel("y")
legend("Euler","ODE45")

subplot(3,2,5)
plot(t,X_Euler(:,3),'-b')
hold on
plot(t,X_ODE45(:,3),'-r')
grid on
grid minor
xlabel("t")
ylabel("z")
legend("Euler","ODE45")

subplot(3,2,[2 4 6])
plot3(X_Euler(:,1),X_Euler(:,2),X_Euler(:,3),'-b')
hold on
plot3(X_ODE45(:,1),X_ODE45(:,2),X_ODE45(:,3),'-r')
grid on
grid minor
xlabel("x")
ylabel("y")
zlabel("z")
legend("Euler","ODE45")
view([30 30])





