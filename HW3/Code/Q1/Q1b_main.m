clear;
close all;
clc;

% Reading the image
img = double(imread('barbara256.png'));
figure; imagesc(img); colormap(gray); title('Original Image');

% Adding Gaussian noise with mean 0 and variance 3 to the image
noisy_img = img + sqrt(3)*randn(size(img));
figure; imagesc(noisy_img); colormap(gray); title('Noisy Image');

% Now we reconstruct img from noisy_img using the prior information that img
% patches have sparse representation in 2D DCT basis

patch_size = 8;                     % Patch size given in the question
reconstructed_image = reconstruct_img(noisy_img, patch_size);

% Displaying the reconstructed image
figure; imagesc(reconstructed_image); colormap(gray); title('Reconstructed Image');

% Displaying the RMSE between the original and reconstructed image
RMSE = norm(img(:) - reconstructed_image(:))/norm(img(:));
fprintf('RMSE = %f\n', RMSE);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%              FUNCTIONS          %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function reconstructed_image = reconstruct_img(img, patch_size)
    % This function computes the compressive measurements of the image patches
    % Inputs:
    %   img: the image
    %   patch_size: the size of the patches
    % Outputs:
    %   reconstructed_image: the reconstructed image

    % Initializing the output variables
    [M, N] = size(img);

    % Initializing the output variables
    reconstructed_image = zeros(M, N);
    img_counts = zeros(M, N);
    
    % Generate the random projection matrix
    phi_ = randn(32, patch_size^2);

    for i = floor(patch_size/2)+1: M-floor(patch_size/2)+1
        disp(i);
        for j = floor(patch_size/2)+1:N-floor(patch_size/2)+1
            patch = img(i-floor(patch_size/2):i+floor(patch_size/2) - 1, j-floor(patch_size/2):j+floor(patch_size/2) - 1);
            vect_patch = patch(:);
            
            % Compute the compressive measurements
            y = phi_*vect_patch;

            % 2D DCT basis matrix
            DCT_basis = dctmtx(patch_size);
            DCT_basis_2D = kron(DCT_basis, DCT_basis);

            % Compute the A matrix
            A = phi_ * DCT_basis_2D;

            theta_estimate = zeros(patch_size^2, 1);    % Initialize the estimate
            lambda = 1;                                % Initialize the lambda
            num_iter = 500;
            theta_estimate = ISTA(A, y, lambda, theta_estimate, num_iter); 
            
            % Compute the estimated patch
            patch_estimate = DCT_basis_2D * theta_estimate;
            patch_estimate = reshape(patch_estimate, [patch_size, patch_size]);

            % Update the reconstructed image
            reconstructed_image(i-floor(patch_size/2):i+floor(patch_size/2) - 1, j-floor(patch_size/2):j+floor(patch_size/2) - 1) = ...
                reconstructed_image(i-floor(patch_size/2):i+floor(patch_size/2) - 1, j-floor(patch_size/2):j+floor(patch_size/2) - 1) + ...
                patch_estimate;

            % Update the image counts
            img_counts(i-floor(patch_size/2):i+floor(patch_size/2) - 1, j-floor(patch_size/2):j+floor(patch_size/2) - 1) = ...
                img_counts(i-floor(patch_size/2):i+floor(patch_size/2) - 1, j-floor(patch_size/2):j+floor(patch_size/2) - 1) + 1;
        end
    end
    reconstructed_image = reconstructed_image ./ img_counts;
end
