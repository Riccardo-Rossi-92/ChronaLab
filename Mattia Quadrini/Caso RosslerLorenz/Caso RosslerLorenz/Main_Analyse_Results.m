clear; clc;

C = 0:0.1:4;

C = [0 0.2 1 1.5 2 2.4];

Case = "RosslerLorenz";

for j = 1 : length(C)


    %% load data
    filename = sprintf('%s_C%.2f.mat', Case, C(j));
    load(filename,"parameters","dlXp","dlXn","dlYp","dlYn","AE");

    %% Predict
    AE_test = AE;
    AE_test.p_interaction_out = 0;
    [dlXpred,dlYpred] = IAE_Network(dlXp,dlYp,1,parameters,AE_test);

    err_X_with_Y = double(extractdata(gather(mean(dlXpred-dlXn,1)))).^2;
    err_Y_with_X = double(extractdata(gather(mean(dlYpred-dlYn,1)))).^2;

    AE_test.p_interaction_out = 1;
    [dlXpred,dlYpred] = IAE_Network(dlXp,dlYp,1,parameters,AE_test);

    err_X_alone = double(extractdata(gather(mean(dlXpred-dlXn,1)))).^2;
    err_Y_alone = double(extractdata(gather(mean(dlYpred-dlYn,1)))).^2;

    %% Analyse data

    figure(1)
    subplot(3,3,min(j,9))
    hold off
    histogram(log10(err_X_alone))
    hold on
    histogram(log10(err_X_with_Y))

    figure(2)
    subplot(3,3,min(j,9))
    hold off
    histogram(log10(err_Y_alone))
    hold on
    histogram(log10(err_Y_with_X))


    S_X_alone = sort(log10(err_X_alone));
    S_X_with_Y = sort(log10(err_X_with_Y));
    S_Y_alone = sort(log10(err_Y_alone));
    S_Y_with_X = sort(log10(err_Y_with_X));

    figure(3)
    subplot(3,3,min(j,9))
    hold off
    plot(S_X_alone,'-')
    hold on
    plot(S_X_with_Y,'-')

    figure(4)
    subplot(3,3,min(j,9))
    hold off
    plot(S_Y_alone,'-')
    hold on
    plot(S_Y_with_X,'-')

    %% Indicators

    p1_X_cause_Y(j) = mean(S_X_alone>S_X_with_Y)/0.5-1;
    p1_Y_cause_X(j) = mean(S_Y_alone>S_Y_with_X)/0.5-1;

    p2_X_cause_Y(j) = mean(S_X_alone-S_X_with_Y);
    p2_Y_cause_X(j) = mean(S_Y_alone-S_Y_with_X);



    % ks test
    [h_ks_y_cause_x(j), p_ks_y_cause_x(j)] = kstest2(S_X_alone,S_X_with_Y);
    [h_ks_x_cause_y(j), p_ks_x_cause_y(j)] = kstest2(S_Y_alone,S_Y_with_X);

end

figure(5)
clf 
subplot(2,2,1)
plot(C,p1_Y_cause_X,'.-','MarkerSize',16,'LineWidth',1.2)
hold on
plot(C,p1_X_cause_Y,'.-','MarkerSize',16,'LineWidth',1.2)

subplot(2,2,2)
plot(C,p2_Y_cause_X,'.-','MarkerSize',16,'LineWidth',1.2)
hold on
plot(C,p2_X_cause_Y,'.-','MarkerSize',16,'LineWidth',1.2)

