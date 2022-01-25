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

% Cost function

% theta = [theta 0; theta 1];
% h_theta = theta0 + Theta1X
% add a column of X_0= 1
%X = [ones(size(X)) X];
h_theta = X * theta;
%theta(1) = 0;
J = ((1 ./ (2.*m) ) .* (sum((h_theta - y).^2))) + ((lambda./ (2.*m)) .* (sum(theta(2:end).^2)));

% % Gradient
% n = size(theta,1);
% %grad_0 = (1./m) .* (sum((h_theta - y) .* X(:,1)));
% for dim = 1:n
% grad(n) = ((1./m) .* (sum((h_theta - y) .* X(:,n)))) + ((lambda ./ m) .* theta(n)) ;
% end

% imported from ex2
grad_0 = 1./m .* sum((h_theta - y) .* X(:,1));
grad_1_to_n = (1./m .* sum((h_theta - y) .* X(:,2:end),1))' + ((lambda .* theta(2:end)) ./ m);

grad = [grad_0; grad_1_to_n];

% =========================================================================

grad = grad(:);

end
