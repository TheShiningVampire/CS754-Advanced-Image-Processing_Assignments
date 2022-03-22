function x = soft(y, lambda)
    % soft function
    % Input:
    %   y: N*1
    %   lambda: scalar
    % Output:
    %   x: N*1

    x = zeros(size(y));
    for i=1:length(y)
        if y(i)>=lambda
            x(i) = y(i)-lambda;
        elseif y(i)<=-lambda
            x(i) = y(i)+lambda;
        else
            x(i) = 0;
        end
end
