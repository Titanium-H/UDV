rng(0,'twister')

for MC = 1:20

r = 3;  % rank of the ground truth
n = 100;    % problem size
d = 3*r*n;   % number of measurements
inds = sort(randperm(n^2,d));   % indexes of the observed entries
A = @(x) A_oper(x,inds);    % handle for the measurement map
AT = @(y) AT_oper(y,n,inds);    % handle for its adjoint

U_true = orth(randn(n,r)); % also try: abs(orth(randn(n,3)));
D_true = diag(1:r);
X_true = U_true*D_true*U_true'; X_true = 0.5*(X_true+X_true');

b = X_true(inds);   % observations

%% Set initial point (to share between all methods)

U0 = randn(n,n);
D0 = eye(n);

%% Save data
if ~exist('data','dir'), mkdir('dir'); end
save(['data/SyntheticData_MC',num2str(MC),'.mat'])

end

%% Functions to be used

function out = A_oper(x,inds)
out = x(inds);
end

function out = AT_oper(y,n,inds)
out = zeros(n,n);
out(inds) = y;
end
