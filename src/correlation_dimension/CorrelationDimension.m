function [corDim,info] = CorrelationDimension(X,method)

%%

if nargin == 1
    method = "MATLAB";
end

%%

if method == "MATLAB"
    corDim = correlationDimension(X);
elseif method == "Krakovska"
    corDim = CorrelationDimension_Krakosvka(X);
end


end