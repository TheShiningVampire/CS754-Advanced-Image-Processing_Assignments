clear;
close all;
clc;

addpath(genpath('MMread'));             % Add MMread to path since mmread is not an inbuilt function since Matlab R2014b
T = 3;                                  % time horizon 

%% Part a
% Reading the cars.avi video, converting to grapscale and extracting the first three frames

% video = mmread('../videos/cars.avi',1:T);
% video_frames_gray = [];                 % Preallocating the video frames
% 
% % Converting to grapscale
% for i = 1:T
%     video_frames_gray = cat(3,video_frames_gray,im2double(rgb2gray(video.frames(i).cdata)));
% end
% 
% % Displaying the first three frames
% % figure; imshow(video_frames_gray(:,:,1)); title('Frame 1'); 
% % figure; imshow(video_frames_gray(:,:,2)); title('Frame 2'); 
% % figure; imshow(video_frames_gray(:,:,3)); title('Frame 3'); 
% 
% video_frames_gray = video_frames_gray(end-120 : end, end-240:end, :);

T = 3;
cars = mmread('../videos/cars.avi', 1:T);
H = cars.height;
W = cars.width;
E = zeros(H, W, T);
for t=1:T
    E(:,:,t) = im2double(rgb2gray(cars.frames(t).cdata));
end
E = E(end-120:end,end-240:end,:);
video_frames_gray = E;

%% Part b
% % Generate a random binary pattern of size HxWxT with elements from {0,1} and compute coded snapshot
% H = size(video_frames_gray, 1);                                          % Height of the video
% W = size(video_frames_gray, 2);                                          % Width of the video
% random_pattern = randi([0 1], H, W, T);                                  % Generate the radom binary pattern
% coded_snapshot = sum(random_pattern .* video_frames_gray, 3);            % Compute the coded snapshot
% noisy_snapshot = coded_snapshot + (0/255)*randn(size(coded_snapshot));   % Adding noise of standard deviation 2

E = E(end-120:end,end-240:end,:);
H = size(E, 1);
W = size(E, 2);

S = randi([0,1], H, W, T);
I = S .* E;
I = sum(I, 3);
sigma = 2;
N = normrnd(0, sigma/255, H, W);
I_plus_N = I + N;

noisy_snapshot = I_plus_N;
random_pattern = S;

% Displaying the Coded Snapshot
figure; imshow(noisy_snapshot); title('Noisy Coded Snapshot');


% Reconstructing the video from the coded snapshot
% reconstructed_video = OMP_reconstruction(noisy_snapshot, random_pattern, T, 5, 0.1);
reconstructed_video = omp_video_reconstruction(noisy_snapshot, random_pattern, T, 8, 0.1);






%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%                     FUNCTIONS                    %%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function I_reconstructed = omp_video_reconstruction(I, S, patch_size, T, epsilon)
    p = patch_size;
    dct_basis = dctmtx(p);
    dct_basis_2D = kron(dct_basis, dct_basis);
    psi = kron(eye(T), dct_basis_2D);
    I_reconstructed = zeros(size(I,1), size(I,2), T);
    counts = zeros(size(I,1), size(I,2), T);
    for i=1:size(I,1) - p + 1
        for j=1:size(I,2) - p + 1
            patch = I(i:i+p-1, j:j+p-1);
            patch_S = S(i:i+p-1, j:j+p-1,:);
            phi = [];
            for t=1:size(S, 3)
                patch_St = patch_S(:, :, t);
                phi_t = diag(reshape(patch_St, p*p, 1));
                phi = [phi phi_t];
            end
            A = phi * psi';
            y = reshape(patch, p*p, 1);
            [theta, ~] = omp(y, A, epsilon);
            i_t = psi' * theta;
            for k=1:T
                f = reshape(i_t((k-1)*p*p+1:k*p*p,:), p, p);
                I_reconstructed(i:i+p-1, j:j+p-1,k) = I_reconstructed(i:i+p-1, j:j+p-1,k) + f;
                counts(i:i+p-1, j:j+p-1,k) = counts(i:i+p-1, j:j+p-1,k) + ones(p,p);
            end
%            disp([i, j]);
        end
    end
    I_reconstructed = I_reconstructed ./ counts;
end

function [theta_opt, T] = omp(y, A, epsilon)
    % Orthogonal Matching Pursuit
    r = y;
    theta = zeros(size(A, 2), 1);
    i = 0;
    T = [];
    A_N = normc(A);
    while (norm(r) > epsilon & i < 64)
        [~, j] = max(abs(r' * A_N));
        T = [T j];
        i = i + 1;
        A_T_i = A(:, T);
        theta(T) = pinv(A_T_i) * y;
        r = y - A_T_i * theta(T);
    end
    theta_opt = theta;
end



% 
% function reconstructed_video = OMP_reconstruction(coded_snapshot, random_pattern, T, patch_size, omp_threshold)
%     % OMP_reconstruction: OMP reconstruction of the coded snapshot
%     %
%     % Inputs:
%     %   coded_snapshot: The coded snapshot
%     %   random_pattern: The random binary pattern
%     %   T: The time horizon
%     %
%     % Outputs:
%     %   reconstructed_video: The reconstructed video sequence
% 
%     
%     % Generating the 2D DCT basis
%     DCT_basis = dctmtx(patch_size);                             % 1D DCT basis
%     psi = kron(DCT_basis', DCT_basis');                         % 2D DCT basis (p^2 x p^2)
%     
%     % Preallocating the reconstructed video
%     reconstructed_video = zeros(size(random_pattern));
% 
%     patch_count = zeros(size(random_pattern, 1), size(random_pattern, 2));
%     w = floor(patch_size/2);                                     % patch size = 2w+1
%     for i = w+1:size(coded_snapshot,1)-w
%         for j = w+1:size(coded_snapshot,2)-w
%             % Extracting the patch
%             image_patch = coded_snapshot(i-w:i+w, j-w:j+w);
%             random_patch = random_pattern(i-w:i+w, j-w:j+w, :);
%             
%             % Computing y
%             y = reshape(image_patch, patch_size^2, 1);                              % image vector (p^2 x1)
% 
%             % Computing the A matrix
%             % A = [];
%             phi = [];
%             for t = 1:T
%                 phi_t = diag(reshape(random_patch(:,:,t), patch_size^2, 1));        % phi_t (p^2 x p^2)
%                 % A_t = phi_t * psi;                                                  % A_t (p^2 x p^2)
%                 % A = cat(2, A, A_t);                                                 % A (p^2 x p^2 T)
%                 phi = cat(2, phi, phi_t);                                            % phi (p^2 x p^2 T)
%             end
% 
%             A = phi* kron(eye(T), psi);                                                  % A (p^2 x p^2 T)
%             % Now we have to estimate theta (p^2 T x 1) from A (p^2 T x p^2) and y (p^2 x 1)
%             theta = Orthogonal_matching_pursuit(A, y, omp_threshold);                % theta (p^2 T x 1)
%             recovered_frames = kron(eye(T), psi) * theta;                            % f (p^2 Tx 1)
%             % recovered_image_patch = reshape(recovered_frames, patch_size^2, T);      % f (p^2 x T)
%             
%             for t = 1:T
%                 recovered_image_patch = reshape(recovered_frames((t-1)*patch_size^2+1:t*patch_size^2,:), patch_size, patch_size);
%                 reconstructed_video(i-w:i+w, j-w:j+w, t) = reconstructed_video(i-w:i+w, j-w:j+w, t) + recovered_image_patch; 
%                 % reshape(recovered_image_patch(:,t), patch_size, patch_size);
% 
%                 patch_count(i-w:i+w, j-w:j+w) = patch_count(i-w:i+w, j-w:j+w) + ones(patch_size, patch_size);
%             end
%         end
%     end
%     reconstructed_video = reconstructed_video ./ patch_count;
% end
% 
% 
% function theta = Orthogonal_matching_pursuit(A, y, threshold)
%     % Orthogonal_matching_pursuit: Orthogonal matching pursuit algorithm
%     %
%     % Inputs:
%     %   A: The A matrix
%     %   y: The y vector
%     %
%     % Outputs:
%     %   theta: The theta vector
%     
%     % Preallocating the theta vector
%     theta = zeros(size(A,2),1);
%     
%     % Initializing the residual
%     residual = y;
%     
%     % Normalizing columns of A
%     A = normc(A);
% 
%     % Initializing the iteration
%     itr = 1;
% 
%     % Support set
%     T = [];
%     
%     % Iterating until the error is less than the threshold
%     while (norm(residual) > threshold && itr < size(A,2))
%         [~, index] = max(abs(residual' * A));          
%         T = [T index];
%         A_T = A(:, T);
%         theta(T) = pinv(A_T) * y;
%         residual = y - A_T * theta(T);
%         itr = itr + 1;
%     end
% end
% 
% 

