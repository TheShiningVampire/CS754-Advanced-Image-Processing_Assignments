clear;
close all;
clc;

% Reading the image
img = double(imread('barbara256.png'));
figure; imagesc(img); colormap(gray); title('Original Image');

% Adding Gaussian noise with mean 0 and variance 3 to the image
noisy_img = img + sqrt(3)*randn(size(img));
figure; imagesc(noisy_img); colormap(gray); title('Noisy Image');

% 2D DCT basis matrix
DCT_basis = dctmtx(8);
DCT_basis_2D = kron(DCT_basis, DCT_basis);

% Now we reconstruct img from noisy_img using the prior information that img patches have sparse representation in 2D DCT basis






%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%             FUNCTIONS       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
    measured_patches = [];
    phi = [];

    for i = floor(patch_size/2)+1:M-floor(patch_size/2)
        for j = floor(patch_size/2)+1:N-floor(patch_size/2)
            patch = img(i-floor(patch_size/2):i+floor(patch_size/2) - 1, j-floor(patch_size/2):j+floor(patch_size/2) - 1);
            vect_patch = patch(:);
        
            % Generate a random matrix phi
            phi_ = randn(32, 64);

            % Compute the compressive measurements
            y = phi_*vect_patch;

            % Append the compressive measurements to the output variable
            measured_patches = [measured_patches, y];

            % Also store the phi matrices
            phi = [phi; phi_];
        end
    end
    
end
