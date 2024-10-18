% Image Recovery (BM and UDU)
clc;
clear;
close all;
rng(0,'twister') % fix the random seed

%% Step 1: Problem setup
% Read images as double type in grayscale, and record image dimensions
img = imread('00_origImage.jpeg');
grayImg = double(rgb2gray(img));
[rows_orig, cols_orig] = size(grayImg); 

% Image preprocessing (Depends on the measurements setting)
measurements_setup = 'Gaussian_1'; % Initialization of A: Gaussian_1, Fourier_1, 
x0 = pre_process_image(measurements_setup, grayImg); % Convert the image into a vector

% Extract necessary info to generate measurement / algorithm
n = size(x0, 1);                     % Dimension of the unknown vector
m = ceil(2*n);                      % Number of measurements (m >= 2n)
r = n;                                      % Rank for the low-rank factorization


%% Step 2: Initialize measurements A and generate measurements y_i = |a_i^T x0|^2
[A, y]  = A_generator(measurements_setup, x0, n, m, rows_orig, cols_orig);


%% Step 3:  Algorithm pre-setting
% Initialize U randomly
U0 = randn(n, r);                        % Note n = r
U0 = U0 ./ norm(U0, 'fro');      % Normalization (Frobenius norm of the resulting matrix is 1)

% Compute Lipschitz constant L (and the step-size alpha=1/L)
L = norm(A * A');                       % Also the largest singular value of the matrix AA'; % L = svds(A * A', 1);
step_size = 1 / L;

% Step 4: Gradient descent parameters
max_iters =1000000;     % Maximum number of iterations

iter_save = generate_list(max_iters);        % Save iteration results

UDU_cell = cell(1, length(iter_save));
udu_cell = cell(1, length(iter_save));
Sds_cell = cell(1, length(iter_save));

UU_cell = cell(1, length(iter_save));
uu_cell = cell(1, length(iter_save));
Ss_cell = cell(1, length(iter_save));


%% Step 4: UDU algorithm
% Gradient descent loop UDU
losses_UDU = zeros(max_iters, 1);    % Record losses
alpha = norm(sqrt(y))/2;                        % norm(sqrt(y)) is an estimation of norm(x0)

U_udu = U0;
D_udu = eye(r);                                         % Identity matrix of size r x r,

cell_index = 1;
for iter = 1:max_iters
    % Compute loss and gradient
    [loss, gradU, gradD] = phase_retrieval_loss_UDU(U_udu, D_udu, A, y, m);  

    U_udu = U_udu - step_size * gradU;        % Gradient update step
    nrmU = norm(U_udu, 'fro');
    if nrmU > alpha, U_udu = U_udu .* (alpha/nrmU); end
    D_udu = D_udu - step_size * gradD;          % Gradient update step
    D_udu = diag(max(diag(D_udu),0));
    losses_UDU(iter) = loss;                                % Store loss for plotting
    
     % Save iteration udu_recovered
    if ismember(iter, iter_save)                           
        [UDU_r, udu_r, Sds] = recovered_udu_image(U_udu, D_udu, x0);
        UDU_cell{cell_index} = UDU_r;
        udu_cell{cell_index} = udu_r;
        Sds_cell{cell_index} = Sds;
        cell_index = cell_index + 1;
    end
end


%% Step 5: UU algorithm: Burer-Monteiro Formulation (X = U * U')
% Gradient descent loop UU
losses_UU = zeros(max_iters, 1);

U_uu = U0;

cell_index = 1;
for iter = 1:max_iters
    % Compute loss and gradient
    [loss, gradU] = phase_retrieval_loss_UU(U_uu, A, y, m);  
    U_uu = U_uu - step_size * gradU;         % Gradient update step
    losses_UU(iter) = loss;                              % Store loss for plotting
    
    % Save iteration udu_recovered
    if ismember(iter, iter_save) 
        [UU_r, uu_r , Ss] = recovered_uu_image(U_uu, x0);
        UU_cell{cell_index} = UU_r;
        uu_cell{cell_index} = uu_r;
        Ss_cell{cell_index} = Ss;
        cell_index = cell_index + 1;
    end
end

%% Step 6: Results and comparison
% Save cells
save('Recovered_raw.mat', 'UDU_cell', 'udu_cell', 'Sds_cell', 'UU_cell', 'uu_cell', 'Ss_cell', 'losses_UDU', 'losses_UU');

% Plot recovered figures
compare_recovered(udu_cell, uu_cell, iter_save, measurements_setup, img, rows_orig, cols_orig)

% Plot statistical analysis
fprintf('Original Signal Norm: %f\n', norm(x0));
fprintf('Recovered Signal Norm: %f\n', norm(udu_cell{end}));

plot_analysis(measurements_setup, x0, udu_cell{end}, uu_cell{end}, losses_UDU, losses_UU, Sds, Ss)



















%% Functions
%% Function to preprocess image and obtain the ground truth x0
function x0 = pre_process_image(measurements_setup, grayImg)
    switch measurements_setup 
        case 'Gaussian_1'
            % Convert grayscale [0, 255] to grayscale [0, 1] then [-1,1]
            scaledGrayImage = (grayImg - min(grayImg(:))) / (max(grayImg(:)) - min(grayImg(:)));
            scaledGrayImage = 2 * scaledGrayImage - 1; 
            x0 = reshape(scaledGrayImage, [], 1);  
            x0 = x0/norm(x0);              % Normalizes the vector, ensuring that its Euclidean norm is 1
        otherwise
            error('Not defined method');
    end
end

%% Function to return measurement A and ground truth in measurement y
function [A, y]  = A_generator(measurements_setup, x0, n, m, ~, ~)
    switch measurements_setup % Obtain the matrix measurement A
        case 'Gaussian_1'
            % Generate random measurement vectors a_i
            A = randn(n, m);                                  % Sample from Gaussian distribution
            % Normalize each row of A 
            row_norms = sqrt(sum(A.^2, 2));  % Compute the norm of each row
            A = A ./ row_norms;                            % Element-wise division to normalize rows
            y = (A'*x0).^2;                                      % Ground truth in measurements
        otherwise
            error('Measurement is not defined.');
    end
end


%% Function to generate saving list
function iter_save = generate_list(limit) % List [1, 10, 100, ..., limit]
    powers_of_ten = 10.^(0:floor(log10(limit)));    
    if powers_of_ten(end) ~= limit
        iter_save = [powers_of_ten, limit];
    else
        iter_save = powers_of_ten;
    end
end


%% Function to compute the loss (objective) and gradient
function [loss, gradU, gradD] = phase_retrieval_loss_UDU(U, D, A, y, m)
    loss = 0;
    n = size(U,1);          % Dimension of the (square) matrix U
    grad = zeros(n,n); % Record gradients 
    for i = 1:m
        ai = A(:,i);             % column vector
        residual = trace((U' * ai) * (ai' * U * D)) - y(i); % Tr(x'aa'x) = Tr(aa'xx') = Tr(a'xx'a) which is predication
        loss = loss + 0.5 * residual^2;
        grad = grad + residual * (ai * ai');
    end
    gradU = (grad + grad') * U * D; 
    gradD = U' * grad * U;
end


%% Function to compute the loss (objective) and gradient
function [loss, gradU] = phase_retrieval_loss_UU(U, A, y, m)
    loss = 0;
    grad = zeros(size(U));
    for i = 1:m
        ai = A(:,i);
        residual = trace((U' * ai) * (ai' * U)) - y(i);
        loss = loss + 0.5 * residual^2;
        grad = grad + residual * (ai * ai');
    end
    gradU = (grad+grad')*U;
end


%% Function to store recovered vector (UDU)
function [UDU_recovered, udu_recovered, Sds] = recovered_udu_image(U_udu, D_udu, x0)
    UDU_recovered = U_udu*D_udu*U_udu';  % Extract the vector from U
    [Udu, Sds, Vdv] = svd(UDU_recovered);
    udu_recovered = Udu(:,1);                                 % Extract the vector from U
    if norm(udu_recovered - x0) >=  norm(udu_recovered + x0)
        udu_recovered = -udu_recovered;
    end
end


%% Function to store recovered vector (UU), Recover the signal from U
function [UU_recovered, uu_recovered, Ss] = recovered_uu_image(U_uu, x0)
    UU_recovered = U_uu*U_uu';  % Extract the vector from U
    [Uu, Ss, Vv] = svd(UU_recovered);
    uu_recovered = Uu(:,1);  % Extract the vector from U
    if norm(uu_recovered - x0) >=  norm(uu_recovered + x0)
        uu_recovered = -uu_recovered;
    end
end


%% Function to recover image matrix from recovered vector
function image_matrix  = v_to_m(measurements_setup, rows_orig, cols_orig, image_vector)

    switch measurements_setup 
        case 'Gaussian_1'
                image_matrix = reshape(image_vector, rows_orig, cols_orig);
        otherwise
            error('Not defined.');
    end

    image_matrix = image_matrix - min(image_matrix(:));
    image_matrix = image_matrix./max(image_matrix(:));
    image_matrix = uint8(255*abs(image_matrix));
end


%% Function to plot compare recovered image
function compare_recovered(udu_cell, uu_cell, iter_save, measurements_setup, img, rows_orig, cols_orig)
    plot_cols= length(udu_cell) + 1;    

    fig_4 = figure('Visible', 'off'); 
    for cell_index = 1 : plot_cols - 1
        udu_recovered_image = v_to_m(measurements_setup, rows_orig, cols_orig, udu_cell{cell_index});
        subplot(2, plot_cols, cell_index);
        imshow(udu_recovered_image, []); 
        title(['UDU:', num2str(iter_save(cell_index))]);
        
        uu_recovered_image = v_to_m(measurements_setup, rows_orig, cols_orig, uu_cell{cell_index});
        subplot(2, plot_cols, plot_cols + cell_index);
        imshow(uu_recovered_image, []);
        title(['UU: ', num2str(iter_save(cell_index))]);
    end
    truth_image = rgb2gray(img);
    
    subplot(2,plot_cols, plot_cols);
    imshow(truth_image, []); 
    title('Ground Truth');
    subplot(2,plot_cols, 2*plot_cols);
    imshow(truth_image, []); 
    title('Ground Truth');

    saveas(fig_4, '04_comparison.jpeg');
end


%% Function to plot statistical analysis
function plot_analysis(~, x0, udu_recovered, uu_recovered, losses_UDU, losses_UU, Sds, Ss)

    % Fit the truth
    fig_1 = figure('Visible', 'off');
    plot(x0, 'k');
    hold on;
    plot(udu_recovered, 'b');
    plot(uu_recovered, 'r');
    title('Recovered Signal');
    xlabel('Index');
    ylabel('Value');
    legend('Truth: x0', 'Recovered: UDU', 'Recovered: UU')
    saveas(fig_1, '01_recovered_signal_figure.jpeg');  
    close(fig_1);
    
    % Losses
    fig_2 = figure('Visible', 'off');
    loglog(losses_UDU(1:end), 'b');  % Plot UDU losses in blue
    hold on;
    loglog(losses_UU(1:end), 'r');   % Plot UU losses in red
    title('Loss over iterations');
    xlabel('Iteration');
    ylabel('Loss');
    legend('UDU', 'UU')
    saveas(fig_2, '02_loss_over_iterations.jpeg');
    close(fig_2);
    
    % Spectrum
    fig_3 = figure('Visible', 'off');
    loglog(diag(Sds), 'b');  % Plot spectrum from Sds in blue
    hold on;
    loglog(diag(Ss), 'r');   % Plot spectrum from Ss in red
    title('Spectrum');
    xlabel('Index');
    ylabel('Value');
    legend('Sds Spectrum', 'Ss Spectrum', 'LineWidth', 2)
    saveas(fig_3, '03_spectrum_plot.jpeg'); 
    close(fig_3);
end
