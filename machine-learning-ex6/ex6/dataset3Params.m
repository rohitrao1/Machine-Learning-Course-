function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

combos = [.01;.03;.1;.3;1;3;10;30];
[C2, sigma2] = meshgrid(combos, combos);
pairs = [C2(:) sigma2(:)];
err = ones(size(pairs,1), 1);

for i = 1:length(err),
  testC = pairs(i,1);
  testSigma = pairs(i,2);
  model = svmTrain(X, y, testC, @(x1,x2) gaussianKernel(x1, x2, testSigma));
  predictions = svmPredict(model, Xval);
  err(i) = mean(double(predictions ~= yval));
end

[minErr, index] = min(err,[],1);

C = pairs(index,1);
sigma = pairs(index, 2);




% =========================================================================

end
