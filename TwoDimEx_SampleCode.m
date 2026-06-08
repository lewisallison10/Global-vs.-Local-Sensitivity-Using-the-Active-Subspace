%% SAMPLE CODE: 2-D Examples

% Sample code for three two-dimensional example models from the paper "Using 
% active subspaces to explore discrepancies between global and local
% parameter sensitivities in a Lotka-Volterra system"

% This code computes subspace distances between global and local active
% subspaces across a grid and generates heatmaps based on those distances.
% It also computes the activity scores from each local region and reports
% how often each rank order occurs.

% Last revised by Huiyan Zou, April 2026

clear all;

% Define the functions, gradients, and names
functions = {
    @(x) exp(0.7 * x(1) + 0.3 * x(2)), ...
    @(x) x(1) * exp(0.7 * x(1) + 0.3 * x(2)), ...
    @(x) exp(0.7 * x(1)^2 * x(2) + 0.3 * x(2))
};

gradients = {
    @(x) [0.7 * exp(0.7 * x(1) + 0.3 * x(2)); ...
          0.3 * exp(0.7 * x(1) + 0.3 * x(2))], ...
    @(x) [exp(0.7 * x(1) + 0.3 * x(2)) * (1 + 0.7 * x(1)); ...
          x(1) * 0.3 * exp(0.7 * x(1) + 0.3 * x(2))], ...
    @(x) [exp(0.7 * x(1)^2 * x(2) + 0.3 * x(2)) * (2 * 0.7 * x(2) * x(1)); ...
          exp(0.7 * x(1)^2 * x(2) + 0.3 * x(2)) * (0.7 * x(1)^2 + 0.3)]
};

function_names = {
    'f_1', ...
    'f_2', ...
    'f_3', ...
};

% Initialization of parameters 
local_grid_size = 0.01;
num_local_samples = 10; % Number of samples within each local grid
dimension = 2; % Dimension of each model
all_rankings = cell(length(functions), 1);

for k = 1:length(functions)
    f = functions{k};
    grad_f = gradients{k};

    ranks_this_func = []; % List to store all the rankings for the current function

    % Number of samples for Latin Hypercube Sampling
    num_global_samples = 100000;
    global_samples = lhsdesign(num_global_samples, dimension); % Global samples in [0, 1] x [0, 1]

    % Compute global gradient directions
    global_gradient_matrix = zeros(dimension, num_global_samples);

    for j = 1:num_global_samples
        x = global_samples(j, :)'; 
        gradient_at_x = grad_f(x); 
        global_gradient_matrix(:, j) = gradient_at_x;
    end

    % Normalize the global gradient matrix
    global_gradient_matrix = global_gradient_matrix / sqrt(num_global_samples);

    % Compute SVD of the normalized global gradient matrix
    [U_global, S_global, ~] = svd(global_gradient_matrix, 'econ');

    % Extract the squared singular values (equivalent to eigenvalues)
    sigma_squared = diag(S_global).^2;

    % Calculate the activity scores for each parameter
    global_activity_scores = zeros(dimension, 1);
    for i = 1:dimension
        for j = 1:min(dimension, size(U_global, 2))
            global_activity_scores(i) = global_activity_scores(i) + sigma_squared(j) * U_global(i, j)^2;
        end
    end

    % Loop over each local grid to calculate the subspace difference
    local_subspace_differences = [];
    local_samples_list = [];
    ranking_map = zeros(1 / local_grid_size, 1 / local_grid_size); % Initialize a map for ranking
    grid_idx = 1; % Index for ranking map

    for x1_start = 0:local_grid_size:1-local_grid_size
        for x2_start = 0:local_grid_size:1-local_grid_size
            % Generate local samples within the current grid
            local_samples = lhsdesign(num_local_samples, dimension) * local_grid_size + [x1_start, x2_start];

            % Compute local gradient directions
            local_gradient_matrix = zeros(dimension, num_local_samples);
            for j = 1:num_local_samples
                x = local_samples(j, :)'; 
                gradient_at_x = grad_f(x); 
                local_gradient_matrix(:, j) = gradient_at_x;
            end

            % Normalize the local gradient matrix
            local_gradient_matrix = local_gradient_matrix / sqrt(num_local_samples);

            % Compute SVD of the local gradient matrix
            [U_local, S_local, ~] = svd(local_gradient_matrix, 'econ');

            % Calculate subspace difference between global and local active
            % subspace (take the first column since only have two) 
            W1 = U_global(:, 1:1);
            W2 = U_local(:, 1:1);
            subspace_difference = norm(W1 * W1' - W2 * W2');

            % Store the subspace differences for each local sample
            local_subspace_differences = [local_subspace_differences; repmat(subspace_difference, num_local_samples, 1)];
            
            % Calculate local activity scores
            local_activity_scores = zeros(dimension, 1);
            sigma_squared_local = diag(S_local).^2;
            for i = 1:dimension
                for j = 1:dimension
                    local_activity_scores(i) = local_activity_scores(i) + sigma_squared_local(j) * U_local(i, j)^2;
                end
            end

            % Rank the parameters based on activity scores
            [~, local_ranking] = sort(local_activity_scores, 'descend');

            % Store the ranking for this grid
            ranks_this_func = [ranks_this_func; local_ranking(:)'];

            % Encode the ranking order into a unique number for the ranking map
            ranking_code = local_ranking(1) * 10 + local_ranking(2);
            ranking_map(grid_idx) = ranking_code;
            grid_idx = grid_idx + 1;

            % Store the local samples 
            local_samples_list = [local_samples_list; local_samples];
        end
    end

    % Save all rankings for this function
    all_rankings{k} = ranks_this_func;

    % Heat map for averaged subspace differences
    figure;
    axis square;
    set(gca,'fontsize',18);
    hold on;
    imagesc([0 1], [0 1], ranking_map);
    scatter(local_samples_list(:,1), local_samples_list(:,2), 50, local_subspace_differences, 'filled');
    title(['Subspace Differences for ', function_names{k}], 'FontSize', 19.8);
    xlabel('x_1');
    ylabel('x_2');
    colorbar;
    colormap('jet'); 
    caxis([0 1]);
    hold on;

    % Plot the white contour line for ranking changes
    [C, h] = contour(linspace(0,1,size(ranking_map,2)), linspace(0,1,size(ranking_map,1)), ranking_map, [12, 21], 'LineColor', 'w', 'LineWidth', 3, 'LineStyle', '--');
    hold off;

    % Display the activity scores
    disp(['Global activity scores for ', function_names{k}, ':']);
    disp(global_activity_scores);

    % Calculate and display the number of all possible rankings (n!)
    num_possible_rankings = factorial(dimension);
    disp(['Number of all possible rankings for ', function_names{k}, ' is ', num2str(num_possible_rankings), '.']);

    % Unique rankings and their frequencies
    [unique_rankings, ~, ranking_indices] = unique(ranks_this_func, 'rows');
    ranking_counts = accumarray(ranking_indices, 1);
    
    % Sort by frequency (high to low)
    [sorted_counts, sorted_indices] = sort(ranking_counts, 'descend');
    sorted_rankings = unique_rankings(sorted_indices, :);

    % Display the number of unique rankings for this function
    num_unique_rankings = size(unique_rankings, 1);
    disp(['Number of unique rankings for ', function_names{k}, ...
          ' across all grids is ', num2str(num_unique_rankings), '.', newline]);
    
    % Display the frequency of each unique ranking
    disp(['Frequency of each unique ranking for ', function_names{k}, ':']);
    for i = 1:num_unique_rankings
        disp(['Ranking ', mat2str(sorted_rankings(i, :)), ...
              ' appears ', num2str(sorted_counts(i)), ' times.']);
    end
end

% Save the results 
% save('2d_AS');
