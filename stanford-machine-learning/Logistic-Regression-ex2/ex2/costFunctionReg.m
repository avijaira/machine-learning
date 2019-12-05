function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
[m, n] = size(X);

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h = sigmoid(X * theta);

% NOT regularizing the theta(1) parameter, which corresponds to theta_0,
% because its used for the bias term.
J = 1/m * sum(-y .* log(h) - (1 - y) .* log(1 - h)) + lambda / (2 * m) * sum(theta(2:end) .^ 2);

for j = 1:n
    if j == 1
        grad(j) = 1/m * sum((h - y) .* X(:, j));
    else
        grad(j) = 1/m * sum((h - y) .* X(:, j)) + lambda / m * theta(j);
    endif
end

% =============================================================

end