function X = CreateSystem(type)

    if type == "Lorenz"
        [~,X] = CoupledLorenz(0,100,10000);
    elseif type == "Rossler"
        [X,~] = CoupledRossler2(0,100,20000); 
    elseif type == "LorenzUncoupled"
        [~,X] = CoupledLorenz(0,100,10000);
    elseif type == "LorenzCoupled"
        [~,X] = CoupledLorenz(9,100,10000);
    end

end