function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
%   cost of using theta as the parameter for linear regression to fit the
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


% Linear Regression cost function for multivariate case in vectorized form
% (ex1.pdf, page 12):

% for j = 0
predictions = X * theta;
errors = predictions - y;
% Following are equivalent: (errors' * errors) == (errors .^ 2)
sqrErrors = errors' * errors;
J = 1 / (2 * m) * sum(sqrErrors);

% Regularized linear regression gradient for j = 0 (ex5.pdf, page 4)
grad = 1 / m * (X' * errors);

temp = theta;
temp(1) = 0;

% for j >= 1
% Following are equivalent: (temp' * temp) == (temp .^ 2)
J = J + lambda / (2 * m) * sum(temp' * temp);

% Regularized linear regression gradient for j >= 1 (ex5.pdf, page 4)
grad = grad + lambda / m * temp;


% =========================================================================

grad = grad(:);

end
