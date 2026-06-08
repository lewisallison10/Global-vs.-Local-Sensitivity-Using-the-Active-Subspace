%% SAMPLE CODE: Lotka-Volterra Model

% Sample code for Lotka-Volterra model from the paper "Using 
% active subspaces to explore discrepancies between global and local
% parameter sensitivities in a Lotka-Volterra system"

% This code generates samples from 8^6 local regions, and computes the
% local active subspace and activity scores for each region. It also
% computes the global active subspace and activity scores, and computes
% subspace distances between the local and global subspaces as described in
% the paper.  All computations are done with respect to the "total tumor
% burden" primary QoI.


%% First, generate the samples

clear all;

% Define the function and parameter ranges
param_ranges = [0, 1; 0, 1; 0, 1; 0, 1; 0, 1; 0, 1];
local_param_ranges = [0, 1; 0, 1; 0, 1; 0, 1; 0, 1; 0, 1];
m = 6; % number of parameters

% Initial conditions for different ratios
initial_ratios = [9 1];
total_volume = 0.02;

% Placeholder initial ratios for analysis
initial_ratio = initial_ratios(1, :);
S0 = initial_ratio(1) / sum(initial_ratio) * 0.02;
R0 = initial_ratio(2) / sum(initial_ratio) * 0.02;

% Local region settings
local_grid_size = 0.125; % Size of the local grid
num_local_samples = 25; % Number of samples within each local grid
dimension = size(param_ranges, 1);

% Nested for loops for 6D grid traversal
all_G_local_vol = {};
all_QoI_local_vol = [];
local_samples_list = {};
 
count = 0;
for x1_start = local_param_ranges(1, 1):local_grid_size:local_param_ranges(1, 2)-local_grid_size
    for x2_start = local_param_ranges(2, 1):local_grid_size:local_param_ranges(2, 2)-local_grid_size
        for x3_start = local_param_ranges(3, 1):local_grid_size:local_param_ranges(3, 2)-local_grid_size
            for x4_start = local_param_ranges(4, 1):local_grid_size:local_param_ranges(4, 2)-local_grid_size
                for x5_start = local_param_ranges(5, 1):local_grid_size:local_param_ranges(5, 2)-local_grid_size
                    for x6_start = local_param_ranges(6, 1):local_grid_size:local_param_ranges(6, 2)-local_grid_size

                        % Update counter 
                        count = count+1;

                        % Generate local samples within the current grid
                        local_samples = lhsdesign(num_local_samples, dimension) * local_grid_size + ...
                            [x1_start, x2_start, x3_start, x4_start, x5_start, x6_start];

                        % Compute local gradient directions
                        local_gradient_matrix_vol = zeros(dimension, num_local_samples);
                        local_qoi_vol = zeros(1,num_local_samples);

                        for j = 1:num_local_samples
                            x = local_samples(j, :)';
                            [local_gradient_matrix_vol(:, j), local_qoi_vol(j)] = grad_f2(x, S0, R0, local_param_ranges);
                        end

                        % Filter out columns with NaN values
                        valid_local_cols = all(~isnan(local_gradient_matrix_vol), 1);
                        local_gradient_matrix_vol = local_gradient_matrix_vol(:, valid_local_cols);
                        local_qoi_vol = local_qoi_vol(valid_local_cols);
                        local_samples = local_samples(valid_local_cols,:);

                        % If not enough valid samples, regenerate the missing samples
                        while size(local_gradient_matrix_vol, 2) < num_local_samples
                            num_missing_samples = num_local_samples - size(local_gradient_matrix_vol, 2);
                            new_samples = lhsdesign(num_missing_samples, dimension) * local_grid_size + ...
                                [x1_start, x2_start, x3_start, x4_start, x5_start, x6_start];
                            for j = 1:size(new_samples, 1)
                                x = new_samples(j, :)';
                                [new_gradient_vol, new_qoi_vol] = grad_f2(x, S0, R0, local_param_ranges);
                                if ~any(isnan(new_gradient_vol)) % Ensure valid gradients
                                    local_gradient_matrix_vol = [local_gradient_matrix_vol, new_gradient_vol]; % Add new column vector
                                    local_qoi_vol = [local_qoi_vol new_qoi_vol];
                                    local_samples = [local_samples; x(:)'];
                                end
                            end
                        end

                        % Store the gradient matrices
                        all_G_local_vol{end+1} = local_gradient_matrix_vol;

                        % Store the QoIs
                        all_QoI_local_vol(end+1,:) = local_qoi_vol;

                        % Store the local samples
                        local_samples_list{end+1} = local_samples;
                    end
                end
            end
        end
    end
end

%% Now calculate global active subspace and activity scores

M = size(all_G_local_vol{1},2);

% First, compile all local gradient samples into global matrix
G_global_vol = [];
for i = 1:length(all_G_local_vol)
    G_global_vol = [G_global_vol all_G_local_vol{i}];
end

% Compute the eigendecomposition
C_global_vol = (1/size(G_global_vol,2))*(G_global_vol*G_global_vol');
[eigVec_global_vol, eigVal_global_vol] = eigs(C_global_vol,m);

% Compute the activity scores
actScores_global_vol = [];
for j = 1:m
    actScores_global_vol(j) = dot(diag(eigVal_global_vol),eigVec_global_vol(j,:).^2);
end


%% Now compute distances between local and global active subspaces

% Storage for subspace distances and local activity scores
subspace_vol = [];
actScores_local_vol_store = [];

for i = 1:length(all_G_local_vol)

    % Calculate the active subspace of the local region
    G_local_vol = all_G_local_vol{i};
    [eigVec_local_vol, eigVal_local_vol] = eigs((1/M)*(G_local_vol*G_local_vol'),m);


    % Choose dimension of local active subspace
    cumEvals_vol = cumsum(diag(eigVal_local_vol));
    normEnergy_vol = cumEvals_vol/cumEvals_vol(end);
    asDim_local_vol = find(normEnergy_vol>.95,1);

    % Calculate the subspace distance based on local dimension:
    W1_vol = eigVec_global_vol(:, 1:asDim_local_vol);
    W2_vol = eigVec_local_vol(:,1:asDim_local_vol);
    subspace_vol(i) = norm(W1_vol * W1_vol' - W2_vol * W2_vol');

    % Calculate local activity scores for later analysis:
    actScores_local_vol = [];
    for j = 1:m
        actScores_local_vol(j) = dot(diag(eigVal_local_vol),eigVec_local_vol(j,:).^2);
    end

    actScores_local_vol_store = [actScores_local_vol_store; actScores_local_vol];
end


%% Function to compute the gradients
function [volume_gradient, volume_qoi] = grad_f2(params, S0, R0, param_ranges)

% Fixed uniform time vector for integration
t_uniform = 0:1:56;

delta = max(1e-5 * (param_ranges(:, 2) - param_ranges(:, 1)), 1e-8);
gradientsVol = zeros(1, length(params));
gradientsRatio = zeros(1, length(params));

options = odeset('RelTol', 1e-6, 'AbsTol', 1e-8, 'MaxStep', 1);

try
    [~, Y_base] = ode45(@(t, Y) lvODE(t, Y, params), t_uniform, [S0 R0], options);
    total_volume_base = trapz(t_uniform, Y_base(:,1) + Y_base(:,2));
catch
    total_volume_base = NaN;
end

for i = 1:length(params)
    params_plus = params;
    params_plus(i) = params_plus(i) + delta(i);

    if params_plus(i) < param_ranges(i, 1) || params_plus(i) > param_ranges(i, 2)
        gradientsVol(i) = NaN;
        continue;
    end


    try
        [~, Y_plus] = ode45(@(t, Y) lvODE(t, Y, params_plus), t_uniform, [S0 R0], options);
        total_volume_plus = trapz(t_uniform, Y_plus(:,1) + Y_plus(:,2));
    catch
        total_volume_plus = NaN;
    end

    if ~isnan(total_volume_plus) && ~isnan(total_volume_base)
        gradientsVol(i) = (total_volume_plus - total_volume_base) / (delta(i));
    else
        gradientsVol(i) = NaN;
    end
end

volume_gradient = gradientsVol(:);
volume_qoi = total_volume_base;

end


%% ODE function for the Lotka-Volterra system
function dYdt = lvODE(~, Y, params)
    rS = params(1);
    rR = params(2);
    KS = params(3);
    KR = params(4);
    gammaS = params(5);
    gammaR = params(6);

    S = Y(1);
    R = Y(2);

    dSdt = rS * S * (1 - S / KS - gammaR * R / KS);
    dRdt = rR * R * (1 - R / KR - gammaS * S / KR);

    dYdt = [dSdt; dRdt];
end
