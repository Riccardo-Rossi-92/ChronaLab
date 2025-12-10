
function chronalab_init(restore_default_path)

if nargin < 1
    restore_default_path = 0;
end

%% restore paths 

if restore_default_path == 1
    disp("Restoring default MATLAB paths")
    restoredefaultpath
    disp("Default MATLAB paths restored")

end

%%

    % Directory for VirtualLab
    path_main = fileparts(mfilename('fullpath'));

    paths_to_add = ["/src/autoencoder";
                    "/src/correlation_dimension";
                    "/src/generate_time_series";
                    "/src/generate_coupled_systems";
                    "/src/autoencoder_dimension";
                    "/src/embedding";
                    "/examples";
                    "/analyses";
                    "/analyses/DimensionCausalityDetection";
                    "/analyses/Denoising";
                    "/Matteo_Scarcella";
                    "/Matteo_Scarcella/models";
                    "/Matteo_Scarcella/src/autoencoder_matteo";
                    "/Matteo_Scarcella/src/generate_lorenz_system";
                    "/Matteo_Scarcella/src/generate_rossler_system";
                    "/Matteo_Scarcella/src/IDEA";
                    "/Matteo_Scarcella/src/IDEA_matteo";
                    "/Matteo_Scarcella/src/TwoNN";
                    "/Matteo_Scarcella/src/utils"];

    for i = 1 : length(paths_to_add)
        path_new = path_main + paths_to_add(i);
        if ~contains(path, path_new)
            addpath(path_new);
            fprintf('new added path : %s/n', path_new);
        end
    end

end