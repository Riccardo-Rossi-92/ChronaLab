
function [X,Y,CX,CY,parameters] = IAE_Network(X,Y,Predict,parameters,AE)

Layer_En = AE.Layer_En;
Layer_Dec = AE.Layer_Dec;

if Predict == 0

    CodeSize = AE.CodeSize;
    OutputSizeX = size(X,1);
    OutputSizeY = size(Y,1);
    
    %% Initialise net

    parameters = [];

    parameters.scale.Xmean = dlarray(mean(X,2));
    parameters.scale.Xstd = dlarray(std(X,[],2));

    parameters.scale.Ymean = dlarray(mean(Y,2));
    parameters.scale.Ystd = dlarray(std(Y,[],2));

    X = (X - parameters.scale.Xmean)./parameters.scale.Xstd; 
    Y = (Y - parameters.scale.Ymean)./parameters.scale.Ystd; 

    for i = 1 : length(Layer_En)

        parameters.("enx"+i).weights = dlarray(randn([Layer_En(i) size(X,1)])*sqrt(2/Layer_En(i)));
        parameters.("enx"+i).bias = dlarray(zeros([Layer_En(i) 1]));

        X = fullyconnect(X,parameters.("enx"+i).weights,parameters.("enx"+i).bias);
        X = sigmoid(X).*X;

        parameters.("eny"+i).weights = dlarray(randn([Layer_En(i) size(Y,1)])*sqrt(2/Layer_En(i)));
        parameters.("eny"+i).bias = dlarray(zeros([Layer_En(i) 1]));

        Y = fullyconnect(Y,parameters.("eny"+i).weights,parameters.("eny"+i).bias);
        Y = sigmoid(Y).*Y;

    end

    parameters.CX.weights = dlarray(randn([CodeSize(1) size(X,1)]))/3;
    parameters.CX.bias = dlarray(zeros([CodeSize(1) 1]));

    parameters.CY.weights = dlarray(randn([CodeSize(2) size(Y,1)]))/3;
    parameters.CY.bias = dlarray(zeros([CodeSize(2) 1]));

    parameters.Int.XY_w = dlarray(zeros([CodeSize(2) CodeSize(1)]));
    parameters.Int.YX_w = dlarray(zeros([CodeSize(1) CodeSize(2)]));

    parameters.Int.XY_b = dlarray(zeros([CodeSize(2) 1]));
    parameters.Int.YX_b = dlarray(zeros([CodeSize(1) 1]));

    CX = fullyconnect(X,parameters.CX.weights,parameters.CX.bias);
    CY = fullyconnect(Y,parameters.CY.weights,parameters.CY.bias);
    
    X = CX + fullyconnect(CY, parameters.Int.YX_w,parameters.Int.YX_b);
    Y = CY + fullyconnect(CX, parameters.Int.XY_w,parameters.Int.XY_b);

    for i = 1 : length(Layer_Dec)

        parameters.("decx"+i).weights = dlarray(randn([Layer_Dec(i) size(X,1)])*sqrt(2/Layer_Dec(i)));
        parameters.("decx"+i).bias = dlarray(zeros([Layer_Dec(i) 1]));

        X = fullyconnect(X,parameters.("decx"+i).weights,parameters.("decx"+i).bias);
        X = sigmoid(X).*X;

        parameters.("decy"+i).weights = dlarray(randn([Layer_Dec(i) size(Y,1)])*sqrt(2/Layer_Dec(i)));
        parameters.("decy"+i).bias = dlarray(zeros([Layer_Dec(i) 1]));

        Y = fullyconnect(Y,parameters.("decy"+i).weights,parameters.("decy"+i).bias);
        Y = sigmoid(Y).*Y;

    end

    parameters.Outputx.weights = dlarray(randn([OutputSizeX size(X,1)]))/3;
    parameters.Outputx.bias = dlarray(zeros([OutputSizeX 1]));

    parameters.Outputy.weights = dlarray(randn([OutputSizeY size(Y,1)]))/3;
    parameters.Outputy.bias = dlarray(zeros([OutputSizeY 1]));

    X = fullyconnect(X,parameters.Outputx.weights,parameters.Outputx.bias);
    Y = fullyconnect(Y,parameters.Outputy.weights,parameters.Outputy.bias);

elseif Predict == 1

    %% Predict

    X = (X - parameters.scale.Xmean)./parameters.scale.Xstd; 
    Y = (Y - parameters.scale.Ymean)./parameters.scale.Ystd; 

    for i = 1 : length(Layer_En)

        X = fullyconnect(X,parameters.("enx"+i).weights,parameters.("enx"+i).bias);
        X = sigmoid(X).*X;

        Y = fullyconnect(Y,parameters.("eny"+i).weights,parameters.("eny"+i).bias);
        Y = sigmoid(Y).*Y;

    end

    CX = fullyconnect(X,parameters.CX.weights,parameters.CX.bias);
    CY = fullyconnect(Y,parameters.CY.weights,parameters.CY.bias);

    C_on = rand(1,size(CX,2))>AE.p_interaction_out; C_on = abs(C_on);
    
    X = CX + fullyconnect(CY, parameters.Int.YX_w,parameters.Int.YX_b).*C_on;
    Y = CY + fullyconnect(CX, parameters.Int.XY_w,parameters.Int.XY_b).*C_on;

    for i = 1 : length(Layer_Dec)

        X = fullyconnect(X,parameters.("decx"+i).weights,parameters.("decx"+i).bias);
        X = sigmoid(X).*X;

        Y = fullyconnect(Y,parameters.("decy"+i).weights,parameters.("decy"+i).bias);
        Y = sigmoid(Y).*Y;

    end

    X = fullyconnect(X,parameters.Outputx.weights,parameters.Outputx.bias);
    Y = fullyconnect(Y,parameters.Outputy.weights,parameters.Outputy.bias);

end


