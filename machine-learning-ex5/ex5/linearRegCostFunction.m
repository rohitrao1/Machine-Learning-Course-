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
%X = [ones(m,1) X];

%Cost Function
h = X*theta;
errors = h - y;
J = sum(errors .^2)/(2*m); 

theta(1) = 0; 
reg = sum(theta .^2) * lambda/(2*m);
J = J + reg; 

%Gradient 

grad = (X' * errors) .* (1/m); %(nx1) vector 
grad_reg = theta .* (lambda/m);
grad = grad + grad_reg; 

%{
n = size(X,2); 
for j = 2:n,
  grad(j) = (1/m) * sum(error .* X(:,j)) + ((lambda/m) * theta(j));
end 
%}











% =========================================================================

grad = grad(:);

end
