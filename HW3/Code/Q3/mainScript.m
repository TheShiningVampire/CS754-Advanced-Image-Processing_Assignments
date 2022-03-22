clear;
close all;
clc;

% Adding path of the l1_ls package
addpath("./l1_ls_matlab");

%%
% Read the images
slice_50 = imread('slice_50.png');
slice_51 = imread('slice_51.png');

% Pad the images with zeros to make them square
slice_50 = padarray(slice_50, [floor((size(slice_50, 2) - size(slice_50, 1))/2) , 0], 0);
slice_51 = padarray(slice_51, [floor((size(slice_51, 2) - size(slice_51, 1))/2) , 0], 0);

% Display the images
figure; imshow(slice_50); title('Slice 50'); colormap(gray);
figure; imshow(slice_51); title('Slice 51'); colormap(gray);

% Creating measurements
% For parts a & b of this assignment we will use equispaced angles to find the radon transform
n_angles = 18;                              % Number of projection angles     
angles = linspace(0, 179, n_angles);        % 18 equispaced angles between 0 and 179 degrees

radon_50 = radon(slice_50, angles);         % Radon transform of slice 50
radon_51 = radon(slice_51, angles);         % Radon transform of slice 51

%% Part a
% Filtered backprojection using Ram-Lak filter
recon_50 = iradon(radon_50, angles, 'Ram-Lak', 1, 'linear');
recon_51 = iradon(radon_51, angles, 'Ram-Lak', 1, 'linear');

recon_50 = recon_50/max(max(recon_50));     % Normalize the image
recon_51 = recon_51/max(max(recon_51));     % Normalize the image   

% Display the reconstructed images
figure; imshow(recon_50); title('Ram Lak Filter reconstruction (50)'); colormap(gray);
figure; imshow(recon_51); title('Ram Lak Filter reconstruction (51)'); colormap(gray);

%% Part b
% Independent Compressed Sensing based reconstruction

y_50 = radon_50(:);                          % Measurements of slice 50 in vector form
y_51 = radon_51(:);                          % Measurements of slice 51 in vector form

% Hyperparameters for l1_ls optimization
m = size(y_50, 1);                           % Number of measurements
n = size((slice_50(:)), 1);                  % Number of unknowns
lambda = 1;                                  % Regularization parameter
rel_tol = 1e-6;                              % Relative tolerance
quiet = false;                               % Display progress

data_size = size(slice_50, 1);               % Size of the data
measurement_size = size(radon_50, 1);        % Size of the measurements

% Function handle for the objective function
A = forward_model_ind(angles,n_angles, 0, data_size, measurement_size);
A_t = forward_model_ind(angles,n_angles, 1, data_size, measurement_size);

% Performing the optimization
[theta_est_50, status_b_50] = l1_ls(A, A_t, m, n, y_50, lambda, rel_tol, quiet);
[theta_est_51, status_b_51] = l1_ls(A, A_t, m, n, y_51, lambda, rel_tol, quiet);

theta_est_50 = reshape(theta_est_50, [data_size, data_size]);
theta_est_51 = reshape(theta_est_51, [data_size, data_size]);

recon_cs_50 = idct2(theta_est_50);                  % Reconstructing the image using the estimated 
                                                    % DCT coefficients
recon_cs_51 = idct2(theta_est_51);                  % Reconstructing the image using the estimated 
                                                    % DCT coefficients

recon_cs_50 = recon_cs_50/max(max(recon_cs_50));    % Normalize the image
recon_cs_51 = recon_cs_51/max(max(recon_cs_51));    % Normalize the image

% Display the reconstructed images
figure; imshow(recon_cs_50); title('(CS based) Reconstructed slice 50'); colormap(gray);


%% Part c
% Coupled Compressed Sensing based reconstruction
% Note: For this part we will use 18 random angles in [0, pi) to find the radon transform

angles_50 = 179*rand(n_angles, 1);       % random angles between 0 and 179 degrees
angles_51 = 179*rand(n_angles, 1);       % random angles between 0 and 179 degrees

% Creating the measurements
radon_50 = radon(slice_50, angles_50);   % Radon transform of slice 50
radon_51 = radon(slice_51, angles_51);   % Radon transform of slice 51

y_50 = radon_50(:);                      % Measurements of slice 50 in vector form
y_51 = radon_51(:);                      % Measurements of slice 51 in vector form

% Appending the y_50 and y_51 as a 1D column vector
y = [y_50; y_51];

% Hyperparameters for l1_ls optimization
m = size(y, 1);                              % Number of measurements
n = 2*size((slice_50(:)), 1);                % Number of unknowns
lambda = 1;                                  % Regularization parameter
rel_tol = 1e-6;                              % Relative tolerance
quiet = false;                               % Display progress

data_size = size(slice_50, 1);               % Size of the data
measurement_size = size(radon_50, 1);        % Size of the measurements

% Function handle for the objective function
A = forward_model_coupled(angles_50, angles_51, n_angles, 0, data_size, measurement_size);
A_t = forward_model_coupled(angles_50, angles_51, n_angles, 1, data_size, measurement_size);

% Performing the optimization
[theta_est, status_c] = l1_ls(A, A_t, m, n, y, lambda, rel_tol, quiet);

theta_est_50 = theta_est(1:data_size^2);
theta_est_51 = theta_est(1:data_size^2) + theta_est(data_size^2+1:end);

theta_est_50 = reshape(theta_est_50, [data_size, data_size]);
theta_est_51 = reshape(theta_est_51, [data_size, data_size]);

recon_ccs_50 = idct2(theta_est_50);                  
recon_ccs_51 = idct2(theta_est_51);

recon_ccs_50 = recon_ccs_50/max(max(recon_ccs_50));
recon_ccs_51 = recon_ccs_51/max(max(recon_ccs_51));

figure;imshow(recon_ccs_50); title('(Coupled CS based) Reconstructed slice 50'); colormap(gray);
figure;imshow(recon_ccs_51); title('(Coupled CS based) Reconstructed slice 51'); colormap(gray);
