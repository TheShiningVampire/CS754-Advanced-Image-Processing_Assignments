clear;
close all;  
clc;

% Adding path of the l1_ls package
addpath("./l1_ls_matlab");

%% x which is sparse in the canonical basis and contains n elements, which is compressively sensed 
% in the form y = phi x+ n where y, the  measurement vector, has m elements and phi is the m×n sensing 
% matrix

n = 500;
m = 200;
x_0_norm = 18;

phi = zeros(m,n);

% Construction of x
% Draw the non-zero elements of x at randomly chosen location, and let their values be drawn
% randomly from Uniform(0,1000)

non_zero_elements = randperm(n,x_0_norm);
x = zeros(n,1);
x(non_zero_elements) = rand(x_0_norm,1)*1000;


% Entries of phi are chosen randomly from +-Bernoulli(0.5)
for i = 1:m
    for j = 1:n
        phi(i,j) = 2*(rand<0.5)-1;
    end
end

% Generate the measurement vector y_measurement
sigma = 0.05*sum(phi*x)/m;
y_measurement = phi*x + sigma*randn(m,1);

% Constructing the reconstruction set (R) and validation set (V)
R = randperm(m, 0.9*m);
V = setdiff(1:m, R);

R_measurements = y_measurement(R);
V_measurements = y_measurement(V);

phi_R = phi(R,:);       % phi_R is the sensing matrix of R
phi_V = phi(V,:);       % phi_V is the sensing matrix of V

% Defining the possible values of lambda
lambdas = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1, 2, 5, 1e1, 5e1, 1e2, 5e2, 1e3, 5e3];

% Hyperparameters for l1_s optimization
rel_tol = 1e-7;                             % Relative tolerance
quiet = true;                               % Display progress


VEs = [];
RMSEs = [];

for i = 1:length(lambdas)
    lambda = lambdas(i);
    disp(i);
    x_reconstructed = l1_ls(phi_R, phi_R', 0.9*m, n, R_measurements, lambda, rel_tol, quiet);

    % Calculating the VE and RMSE on the validation set
    VE = validation_error(x_reconstructed, phi_V, V_measurements);
    rmse = RMSE(x_reconstructed, x);
    VEs = [VEs, VE];
    RMSEs = [RMSEs, rmse];
end

% Plotting the validation error vs lambda
figure; 
semilogx(lambdas, VEs, '-o');
xlabel('Lambda');
ylabel('Validation Error');
title('Validation Error vs Lambda');
hold on;
% Highlight the minimum VE
[min_VE, min_VE_index] = min(VEs);
plot(lambdas(min_VE_index), min_VE, 'ro');
hold off;

% Plotting the RMSE vs lambda
figure;
semilogx(lambdas, RMSEs, '-o'); 
xlabel('Lambda');
ylabel('RMSE');
title('RMSE vs Lambda');
hold on;
% Highlight the minimum RMSE
[min_RMSE, min_RMSE_index] = min(RMSEs);
plot(lambdas(min_RMSE_index), min_RMSE, 'ro');
hold off;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                   FUNCTIONS               %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function VE = validation_error(x_recon, phi_V, V_measurements)
    % This function computes the validation error of the reconstructed x
    % given the validation set
    % Inputs:
    % x: the reconstructed vector
    % phi_V: the sensing matrix of the validation set
    % V_measurements: the measurements of the validation set
    % Outputs:
    % validation_error: the validation error of the reconstructed x
    VE = norm(phi_V*x_recon - V_measurements)/length(V_measurements);
end

function rmse = RMSE(x_recon, x)
    % This function computes the RMSE of the reconstructed x
    % given the original x
    % Inputs:
    % x: the reconstructed vector
    % x_0: the original vector
    % Outputs:
    % rmse: the RMSE of the reconstructed x
    rmse = norm(x_recon - x)/norm(x);
end



