function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification

%NN will perform classification (classifying handwritten images) 
%This function in particular is meant to calculate the optimal cost and gradient given a set of parameters


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

%modifying y to become a matrix
%Add 9 columns to pre-existing vector. 
%Go through each row and 
c = (1:num_labels);

temp = ones(size(y), num_labels); %5000x10 

%for i = 1:m,
%  temp(i,:) = (y(i) == c); %1x10 row of 0 or 1 --> Goes up to m examples 
%end

%Vectorized solution
temp(1:m,:) = (y(1:m) == c);

%Computing prediction with forward propogation
tempX = X;
X = [ones(m, 1) X]; %Now 5000x401
a2 = sigmoid(Theta1 * X'); %25x5000

len = size(a2,2); 
o = ones(1, len);
a2 = [o;a2]; %Now 26x5000

h = sigmoid(Theta2*a2); %10x5000 --> Each row is 0 or 1 for every example


%Computing cost function w/o regularization
for i = 1:m,
  J = J + sum((-temp(i,:)' .* log(h(:,i))) - ((1-temp(i,:)') .* log(1-h(:,i))));
  %Use sum or not? 
end
J = J/m;


%Adding regularization to cost function
reg = 0;
m1 = size(Theta1,1);
n1 = size(Theta1, 2);
for j = 1:m1,
  term1 = Theta1(j, 2:n1);
  reg = reg + sum(term1 .^ 2); 
end

m2 = size(Theta2, 1);
n2 = size(Theta2, 2);
for k = 1:m2,
  term2 = Theta2(k, 2:n2);
  reg = reg + sum(term2 .^ 2);
end

reg = reg * (lambda/(2*m));

J = J + reg; 

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
a1 = X;
z2 = a1 * Theta1';
a2 = sigmoid(z2);
a2 = [ones(size(a2,1), 1) a2];

z3 = a2 * Theta2';
a3 = sigmoid(z3);

d3 = a3 - temp;
d2 = (d3 * Theta2(:,2:end)) .* sigmoidGradient(z2);

Delta1 = d2' * a1;
Delta2 = d3' * a2;

Theta1_grad = (1/m) * Delta1;
Theta2_grad = (1/m) * Delta2;


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Theta1(:,1) = 0;
Theta2(:,1) = 0; 
Theta1_grad = Theta1_grad + (Theta1 .* (lambda/m));
Theta2_grad = Theta2_grad + (Theta2 .* (lambda/m));















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
