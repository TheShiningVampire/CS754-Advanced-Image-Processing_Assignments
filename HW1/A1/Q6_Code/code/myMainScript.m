clear;
close all;
clc;

addpath(genpath('MMread'));                         % Add MMread to path since mmread is not an inbuilt function since Matlab R2014b

video_name = "cars";                                % Also uncomment line 16
% video_name = "flame";                             % Also uncommnet line 17

%% Part a
% Reading the cars.avi video file and converting to grapscale and extracting the first three frames
T = 3;                                            % time horizon 
% T = 5;                                          % Select the appropriate number of frames
% T = 7;

video = mmread('../videos/cars.avi',1:T);
% video = mmread('../videos/flame.avi', 1:T);

video_frames_gray = read_gray_video(video, T, video_name);      % We have considered only the lower 120x240 pixels of the video
% Displaying the first three frames
display_frames(video_frames_gray, T, 'Original Video Frame ');
save_frames(video_name, T, video_frames_gray, 'orig_');

%% Part b
% Generate the coded snapshot
[noisy_snapshot, random_pattern] = get_coded_snapshot(video_frames_gray, T);    % Get the noisy coded snapshot and the random pattern

% Displaying the Coded Snapshot
figure; imagesc(noisy_snapshot); colormap('gray'); title('Noisy Coded Snapshot');
imwrite(noisy_snapshot, "../results/"+video_name+"_"+"noisy_snapshot_"+num2str(T)+".png");

%% Part e
% Reconstructing the video from the coded snapshot
epsilon = 0.1;                                  % Threshold for the reconstruction error (Calaculation shown in report)
reconstructed_video = OMP_reconstruction(noisy_snapshot, random_pattern, T, 8, epsilon);

% Displaying the Reconstructed Video
display_frames(reconstructed_video, T, 'Reconstructed Video Frame ');
save_frames(video_name, T, reconstructed_video, 'recon_');

% Calculating the RMSE between the reconstructed video and the original video
RMSE_T_frames = RMSE(video_frames_gray, reconstructed_video)                   % RMSE for the first T frames



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%                     FUNCTIONS                    %%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function video_frames_gray = read_gray_video(video, T, video_name)
    % This function reads the video and converts it to grayscale
    % Inputs:
    %   video: Video structure obtained from mmread
    %   T: Number of frames to be read
    % Outputs:
    %   video_frames_gray: Grayscale video frames
    %

    video_frames_gray = [];                              % Preallocating the video frames
    % Converting to grayscale
    for i = 1:T
        video_frames_gray = cat(3,video_frames_gray,im2double(rgb2gray(video.frames(i).cdata)));
    end
    
    if (video_name == 'cars')
        video_frames_gray = video_frames_gray(end-120 : end, end-240:end, :);    
    end
end

function display_frames(video_frames_gray, T, title_text)
    % This function displays the first T frames of the video
    % Inputs:
    %   video_frames_gray: Grayscale video frames
    %   T: Number of frames to be displayed
    % Outputs:
    %   None
    %

    for i = 1:T
        figure; imagesc(video_frames_gray(:,:,i)); colormap('gray'); title(title_text + num2str(i));
    end
end

function save_frames(video_name, T, video_frames_gray, title_text)
    % This function saves the first T frames of the video
    % Inputs:
    %   video_name: Name of the video
    %   T: Number of frames to be saved
    %   video_frames_gray: Grayscale video frames
    % Outputs:
    %   None
    %

    for i = 1:T
        imwrite(video_frames_gray(:,:,i), '../results/'+video_name+'_'+num2str(T)+'_'+title_text+num2str(i)+'.png');
    end
end

function [noisy_snapshot, random_pattern] = get_coded_snapshot(video_frames_gray, T)
    % This function generates the coded snapshot with noise added
    % Inputs:
    %   video_frames_gray: Grayscale video frames
    %   T: Number of frames to be displayed
    % Outputs:
    %   noisy_snapshot: Coded snapshot with noise added
    %

    H = size(video_frames_gray, 1);                                          % Height of the video
    W = size(video_frames_gray, 2);                                          % Width of the video
    random_pattern = randi([0 1], H, W, T);                                  % Generate the radom binary pattern
    coded_snapshot = sum(random_pattern .* video_frames_gray, 3);            % Compute the coded snapshot
    noisy_snapshot = coded_snapshot + (2/255)*randn(size(coded_snapshot));   % Adding noise of standard deviation 2
end



function rmse = RMSE(A, B)
    % Computes the RMSE (Relative Mean Squared Error) between two video sequence A and B
    % 
    % Inputs:
    %  A: The first video sequence (original data)
    %  B: The second video sequence (reconstructed data)
    %
    % Outputs:
    %  rmse: The RMSE between the two video sequences
    %
    rmse = sum(sum(sum((A - B).^2))) / sum((sum(sum(A.^2))));
end


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
    
    l= floor(patch_size/2);                                      % number of pixels on left of central pixel
    r = floor(patch_size/2) - 1 + mod(patch_size, 2);            % number of pixels on right of central pixel
    for i = l+1:size(coded_snapshot,1)-r
        for j = l+1:size(coded_snapshot,2)-r
            % Extracting the patch
            image_patch = coded_snapshot(i-l:i+r, j-l:j+r);
            random_patch = random_pattern(i-l:i+r, j-l:j+r, :);
            
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
                reconstructed_video(i-l:i+r, j-l:j+r, t) =reconstructed_video(i-l:i+r, j-l:j+r, t) + reshape(recovered_image_patch(:,t), patch_size, patch_size);
            end
            patch_count(i-l:i+r, j-l:j+r) = patch_count(i-l:i+r, j-l:j+r) + 1;
        end
    end
    reconstructed_video = reconstructed_video ./ repmat(patch_count, [1,1,T]);
end


function theta = Orthogonal_matching_pursuit(A, y, threshold)
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

    % Normalize the columns of A
    A_n = A ./ repmat(sqrt(sum(A.^2)), size(A,1), 1);
    
    % Iterating until the error is less than the threshold
    while (norm(residual) > threshold && itr < 64)
        [~, index] = max(abs(residual' * A_n));          
        T = [T index];
        A_T = A(:, T);
        theta(T) = pinv(A_T) * y;
        residual = y - A_T * theta(T);
        itr = itr + 1;
    end
end