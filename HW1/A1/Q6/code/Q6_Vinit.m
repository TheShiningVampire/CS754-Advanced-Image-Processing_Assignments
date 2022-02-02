clear;
close all;
clc;

addpath(genpath('MMread'));             % Add MMread to path since mmread is not an inbuilt function since Matlab R2014b
T = 3;                                  % time horizon 

%% Part a
% Reading the cars.avi video, converting to grapscale and extracting the first three frames

video = mmread('../videos/cars.avi',1:T);
video_frames_gray = [];                 % Preallocating the video frames

% Converting to grapscale
for i = 1:T
    video_frames_gray = cat(3,video_frames_gray,im2double(rgb2gray(video.frames(i).cdata)));
end

% Displaying the first three frames
% figure; imshow(video_frames_gray(:,:,1)); title('Frame 1'); 
% figure; imshow(video_frames_gray(:,:,2)); title('Frame 2'); 
% figure; imshow(video_frames_gray(:,:,3)); title('Frame 3'); 

video_frames_gray = video_frames_gray(end-120 : end, end-240:end, :);

%% Part b
% Generate a random binary pattern of size HxWxT with elements from {0,1} and compute coded snapshot
H = size(video_frames_gray, 1);                                          % Height of the video
W = size(video_frames_gray, 2);                                          % Width of the video
random_pattern = randi([0 1], H, W, T);                                  % Generate the radom binary pattern
coded_snapshot = sum(random_pattern .* video_frames_gray, 3);            % Compute the coded snapshot
noisy_snapshot = coded_snapshot + (0/255)*randn(size(coded_snapshot));   % Adding noise of standard deviation 2

% Displaying the Coded Snapshot
figure; imshow(noisy_snapshot); title('Noisy Coded Snapshot');


% Reconstructing the video from the coded snapshot
reconstructed_video = OMP_reconstruction(noisy_snapshot, random_pattern, T, 8, 0.1);

% Displaying the Reconstructed Video
figure; imshow(reconstructed_video(:,:,1)); title('Reconstructed Video Frame 1');
figure; imshow(reconstructed_video(:,:,2)); title('Reconstructed Video Frame 2');
figure; imshow(reconstructed_video(:,:,3)); title('Reconstructed Video Frame 3');




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%                     FUNCTIONS                    %%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function rmse = RMSE

function reconstructed_video = OMP_reconstruction(coded_snapshot, random_pattern, T, patch_size, omp_threshold)
    % OMP_reconstruction: OMP reconstruction of the coded snapshot
    %
    % Inputs:
    %   coded_snapshot: The coded snapshot
    %   random_pattern: The random binary pattern
    %   T: The time horizon
    %
    % Outputs:
    %   reconstructed_video: The reconstructed video sequence

    
    % Generating the 2D DCT basis
    DCT_basis = dctmtx(patch_size);                             % 1D DCT basis
    psi = kron(DCT_basis', DCT_basis');                         % 2D DCT basis (p^2 x p^2)
    
    % Preallocating the reconstructed video
    reconstructed_video = zeros(size(random_pattern));

    patch_count = zeros(size(random_pattern, 1), size(random_pattern, 2));
    
    l= floor(patch_size/2);                                     %  
    r = floor(patch_size/2) - 1 + mod(patch_size, 2);            % 
    for i = w+1:size(coded_snapshot,1)-w
        for j = w+1:size(coded_snapshot,2)-w
            % Extracting the patch
            image_patch = coded_snapshot(i-w:i+w, j-w:j+w);
            random_patch = random_pattern(i-w:i+w, j-w:j+w, :);
            
            % Computing y
            y = reshape(image_patch, patch_size^2, 1);                              % image vector (p^2 x1)

            % Computing the A matrix (mentioned by Prof. on Moodle)
            A = [];
            for t = 1:T
                phi_t = diag(reshape(random_patch(:,:,t), patch_size^2, 1));        % phi_t (p^2 x p^2)
                A_t = phi_t * psi;                                                  % A_t (p^2 x p^2)
                A = cat(2, A, A_t);                                                 % A (p^2 x p^2 T)
            end
            % Now we have to estimate theta (p^2 T x 1) from A (p^2 T x p^2) and y (p^2 x 1)
            theta = Orthogonal_matching_pursuit(A, y, omp_threshold);                % theta (p^2 T x 1)
            recovered_frames = kron(eye(T), psi) * theta;                            % f (p^2 Tx 1)
            recovered_image_patch = reshape(recovered_frames, patch_size^2, T);      % f (p^2 x T)
            
            for t = 1:T
                reconstructed_video(i-w:i+w, j-w:j+w, t) =reconstructed_video(i-w:i+w, j-w:j+w, t) + reshape(recovered_image_patch(:,t), patch_size, patch_size);
            end
            patch_count(i-w:i+w, j-w:j+w) = patch_count(i-w:i+w, j-w:j+w) + 1;
        end
    end
    reconstructed_video = reconstructed_video ./ repmat(patch_count, [1,1,T]);
end


function theta_opt = Orthogonal_matching_pursuit(A, y, threshold)
    % Orthogonal_matching_pursuit: Orthogonal matching pursuit algorithm
    %
    % Inputs:
    %   A: The A matrix
    %   y: The y vector
    %
    % Outputs:
    %   theta: The theta vector
    
    % Preallocating the theta vector
    theta = zeros(size(A,2),1);
    
    % Initializing the residual
    residual = y;

    % Initializing the iteration
    itr = 1;

    % Support set
    T = [];
    
    % Iterating until the error is less than the threshold
    while (norm(residual) > threshold && itr < 64)
        [~, index] = max(abs(residual' * normc(A)));          
        T = [T index];
        A_T = A(:, T);
        theta(T) = pinv(A_T) * y;
        residual = y - A_T * theta(T);
        itr = itr + 1;
    end
    theta_opt = theta;
end