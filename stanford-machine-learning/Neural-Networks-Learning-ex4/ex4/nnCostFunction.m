function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Part 1: Feedforward the neural network and return the cost in the variable J.
X = [ones(m, 1) X];    % (m, n+1), where n=400

% Method 1 (per label)
%a1 = X;
%z2 = a1 * Theta1';      % (m, 401) * (401, 25) = (m, 25)
%a2 = sigmoid(z2);       % (m, 25)
%a2 = [ones(m, 1) a2];   % (m, 26)
%z3 = a2 * Theta2';      % (m, 26) * (26, num_labels) = (m, num_labels) = (m, 10)
%a3 = sigmoid(z3);       % (m, num_labels)
%h = a3;                 % (m, num_labels)
%
%for c = 1:num_labels
%    yc = (y == c);      % (m, 1), containing only values 0 or 1, with only y(c) = 1;
%    J = J + 1/m * sum(-yc .* log(h(:, c)) - (1 - yc) .* log(1 - h(:, c)));
%end

% Method 2 (per example)
for t = 1:m
    at1 = X(t, :);          % (1, 401)
    zt2 = at1 * Theta1';    % (1, 25)
    at2 = sigmoid(zt2);     % (1, 25)
    at2 = [ones(1, 1) at2]; % (1, 26)
    zt3 = at2 * Theta2';    % (1, num_labels)
    at3 = sigmoid(zt3);     % (1, num_labels)
    ht = at3;

    for c = 1:num_labels
        yc = (y(t) == c);
        J = J + 1/m * sum(-yc .* log(ht(:, c)) - (1 - yc) .* log(1 - ht(:, c)));
    end
end

% -------------------------------------------------------------
% Part 2: Implement the backpropagation algorithm to compute the gradients.

delta2 = zeros(1, hidden_layer_size + 1);    % hidden_layer_size = 25
delta3 = zeros(1, num_labels);               % num_labels = 10

for t = 1:m
    at1 = X(t, :);          % (1, 401)
    zt2 = at1 * Theta1';    % (1, 401) * (401, 25) = (1, 25)
    at2 = sigmoid(zt2);     % (1, 25)
    at2 = [ones(1, 1) at2]; % (1, 26)
    zt3 = at2 * Theta2';    % (1, 26) * (26, num_labels) = (1, num_labels)
    at3 = sigmoid(zt3);     % (1, num_labels)

    yt = zeros(1, num_labels);
    c = y(t);
    yt(c) = 1;

    delta3 = at3 - yt;      % (1, num_labels)

    delta2 = (delta3 * Theta2) .* sigmoidGradient([ones(1, 1) zt2]);
    delta2 = delta2(2:end);    % (1, 25)

    Theta2_grad = Theta2_grad + delta3' * at2;
    Theta1_grad = Theta1_grad + delta2' * at1;
end

Theta1_grad = (1 / m) * Theta1_grad;
Theta2_grad = (1 / m) * Theta2_grad;

% -------------------------------------------------------------
% Part 3: Implement regularization with the cost function and gradients.

temp1 = Theta1;
temp1(:, 1) = 0;        % (25, 401), set theta corresponding to bias, 1st column, to zero.
J_reg1 = lambda / (2 * m) * sum(sum(temp1 .^ 2));

temp2 = Theta2;
temp2(:, 1) = 0;        % (10, 26), set theta corresponding to bias, 1st column, to zero.
J_reg2 = lambda / (2 * m) * sum(sum(temp2 .^ 2));

J = J + J_reg1 + J_reg2;

% =========================================================================

Theta1_grad_reg = (lambda / m) * temp1;
Theta2_grad_reg = (lambda / m) * temp2;

Theta1_grad = Theta1_grad + Theta1_grad_reg;
Theta2_grad = Theta2_grad + Theta2_grad_reg;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:); Theta2_grad(:)];

end
