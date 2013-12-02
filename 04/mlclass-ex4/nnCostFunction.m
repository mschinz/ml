%I used the following steps:
%Compute delta3 by subtracting yv from a3 - matrix[5000x10]
%Compute intermediate r2 by multiplying delta3 and Theta2 (without first column of θ(2)0 elements) - matrix[5000x25]
%Compute delta2 by per-element multiplication of z2 passed through sigmoidGradient and r2 - matrix[5000x25]
%Compute regularization term t1by multiplying Theta1 by lambda scalar and then setting first column to zero to account for j=0 case - matrix[25x401]
%Compute Theta1_grad (D(2)) by multiplying transposed delta2 and X, adding the regularization term t1 and then dividing by scalar m - matrix[25x401]
%Compute second regularization term t2 in the same manner, but for Theta2 - matrix[10x26]
%Compute Theta2_grad (D(L)) in the same manner as Theta1_grad - matrix[10x26]


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
sig = inline("1.0 ./ (1.0 + exp(-z))");

X = [ones(m, 1) X];
Y = [zeros(m,num_labels)];
for i = 1:m;
	Y(i,y(i,1)) = 1;
end;

z2 = X*Theta1';
a2 = [ones(m,1) sig(z2)];
a3 = sig(a2*Theta2');
left = -Y.*log(a3);
right = (1-Y).*log(1-a3);
J = (1/m)*sum(sum(left-right));

Theta11 = Theta1(:,(2:end));
Theta21 = Theta2(:,(2:end));
reg = lambda/(2*m) * (sum(sumsq(Theta11)) + sum(sumsq(Theta21)));
J = J + reg;

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
delta3 = a3 - Y;
delta2 = (delta3*Theta21).*sigmoidGradient(z2);
t1 = (lambda / m) * Theta1;
t1 (:,1) = 0;
Theta1_grad = (delta2'*X)/m + t1;
t2 = (lambda / m) * Theta2;
t2 (:,1) = 0;
Theta2_grad = (delta3'*a2)/m + t2;
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
