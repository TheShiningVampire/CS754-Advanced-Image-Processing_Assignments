classdef forward_model_coupled
    % FORWARD_MODEL_COUPLED
    %
    %   Class for forward modeling coupled data.

    properties
        angles_1
        angles_2
        n_angles
        transpose
        data_size
        measurement_size
    end

    methods
        function obj = forward_model_coupled(angles_1, angles_2, n_angles, transpose, data_size, ...
             measurement_size)

            obj.angles_1 = angles_1;
            obj.angles_2 = angles_2;
            obj.n_angles = n_angles;
            obj.transpose = transpose;
            obj.data_size = data_size;
            obj.measurement_size = measurement_size;
        end

        % Definition for multiplication with the forward model
        function y = mtimes(A,x)
            if (A.transpose == 0)
                x_1 = reshape(x(1:A.data_size^2), A.data_size, A.data_size);
                x_2 = reshape(x(A.data_size^2+1:end), A.data_size, A.data_size);

                Ux_1 = idct2(x_1);
                R1Ux_1 = radon(Ux_1, A.angles_1);

                R2Ux_1 = radon(Ux_1, A.angles_2);

                Ux_2 = idct2(x_2);
                R2Ux_2 = radon(Ux_2, A.angles_2);

                y_1 = R1Ux_1(:);
                y_2 = R2Ux_1(:) + R2Ux_2(:);

                y = [y_1; y_2];
            else
                y_1 = reshape(x(1:(A.measurement_size* A.n_angles)), A.measurement_size, A.n_angles);
                y_2 = reshape(x((A.measurement_size* A.n_angles)+1:end), A.measurement_size, A.n_angles);


                R1Ty_1 = iradon(y_1, A.angles_1, A.data_size);
                UTR1Ty_1 = dct2(R1Ty_1);

                R2Ty_2 = iradon(y_2, A.angles_2, A.data_size);
                UTR2Ty_2 = dct2(R2Ty_2);

                y_1 = UTR1Ty_1(:) + UTR2Ty_2(:);
                y_2 = UTR2Ty_2(:);

                y = [y_1; y_2];
            end

        end
    end
end
