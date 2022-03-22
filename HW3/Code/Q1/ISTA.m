function theta = ISTA(A, y, lambda, theta_estimate, num_iter)
    % Function to perform ISTA algorithm 
    % Inputs:
    % A: m x n matrix
    % y: m x 1 vector
    % lambda: regularization parameter
    % theta_estimate: initial guess for theta
    % num_iter: number of iterations

    % Output:
    % theta: n x 1 vector (n > m)
    
    eig_vals = eig(A'*A);           % Find the eigenvalues of A'A
    eig_vals = sort(eig_vals);      % Sort the eigenvalues
    alpha = eig_vals(end) + 1;      % Alpha should be greater than the largest 
                                    % eigenvalue

    for i = 1:num_iter
        theta = soft(theta_estimate + (1/alpha)*A'*(y - A*theta_estimate), lambda/(2*alpha));
        theta_estimate = theta;
    end
end
