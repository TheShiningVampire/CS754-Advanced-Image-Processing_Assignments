classdef forward_model_ind
    % FORWARD_MODEL_IND
    %
    %   Class for forward model (A) used in the optimization problem

    properties
        angles
        n_angles
        transpose
        data_size
        measurement_size
    end

    methods
        function obj = forward_model_ind(angles, n_angles, transpose, data_size, measurement_size)
            obj.angles = angles;
            obj.n_angles = n_angles;
            obj.transpose = transpose;
            obj.data_size = data_size;
            obj.measurement_size = measurement_size;
        end

        % Definition for multiplication with the forward model
        function y = mtimes(A,x)
            if (A.transpose == 0)
                x = reshape(x, A.data_size, A.data_size);    % Reshape the input to a matrix (image)
                y = idct2(x);                                % Apply the inverse discrete cosine 
                                                             % transform
                y = radon(y, A.angles);                      % Apply the radon transform
                y = y(:);                                    % Reshape the output to a vector
    
            else
                x = reshape(x, A.measurement_size, A.n_angles);     %Reshape to radon transform 
                                                                    %format
                y = iradon(x, A.angles, A.data_size);               %Apply the inverse radon 
                                                                    %transform 
                y = dct2(y);                                        %Apply the discrete cosine 
                                                                    % transform
                y = y(:);                                           %Reshape to vector

            end

        end
    end
end
