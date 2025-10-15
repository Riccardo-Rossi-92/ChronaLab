clear; clc;

D_target = 3;
N = 10000;

x = sobolset(D_target);
x = x(1:N,:);


