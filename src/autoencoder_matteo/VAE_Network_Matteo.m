function [X,Code,parameters] = VAE_Network(X,Predict,parameters,AE,codePrediction)

if nargin == 4
    codePrediction = 0;
end

Layer_En = AE.Layer_En;
Layer_Dec = AE.Layer_Dec;

if Predict == 0

    CodeSize = AE.CodeSize;
    OutputSize = size(X,1);
    
    %% Initialise net

    parameters = [];

    parameters.scale.Xmean = dlarray(mean(X,2));
    parameters.scale.Xstd = dlarray(std(X,[],2));
    
    X = (X - parameters.scale.Xmean)./parameters.scale.Xstd;

    for i = 1 : length(Layer_En)

        parameters.("en"+i).weights = dlarray(randn([Layer_En(i) size(X,1)])*sqrt(2/Layer_En(i)));
        parameters.("en"+i).bias = dlarray(zeros([Layer_En(i) 1]));

        X = fullyconnect(X,parameters.("en"+i).weights,parameters.("en"+i).bias);
    end

    parameters.Code.weights = dlarray(randn([CodeSize size(X,1)]))/3;
    parameters.Code.bias = dlarray(zeros([CodeSize 1]));

    Code = fullyconnect(X,parameters.Code.weights,parameters.Code.bias);
    X = Code;

    for i = 1 : length(Layer_Dec)

        parameters.("dec"+i).weights = dlarray(randn([Layer_Dec(i) size(X,1)])*sqrt(2/Layer_Dec(i)));
        parameters.("dec"+i).bias = dlarray(zeros([Layer_Dec(i) 1]));

        X = fullyconnect(X,parameters.("dec"+i).weights,parameters.("dec"+i).bias);
    end

    parameters.Output.weights = dlarray(randn([OutputSize size(X,1)]))/3;
    parameters.Output.bias = dlarray(zeros([OutputSize 1]));

    X = fullyconnect(X,parameters.Output.weights,parameters.Output.bias);

elseif Predict == 1

    %% Predict

    X = (X - parameters.scale.Xmean)./parameters.scale.Xstd;

    for i = 1 : length(Layer_En)

        X = fullyconnect(X,parameters.("en"+i).weights,parameters.("en"+i).bias);
        X = tanh(X);

    end

    Code = fullyconnect(X,parameters.Code.weights,parameters.Code.bias);    
    X = Code;

    for i = 1 : length(Layer_Dec)

        X = fullyconnect(X,parameters.("dec"+i).weights,parameters.("dec"+i).bias);
        X = tanh(X);
    end

    X = fullyconnect(X,parameters.Output.weights,parameters.Output.bias);

    X = X.*parameters.scale.Xstd + parameters.scale.Xmean;  

end


