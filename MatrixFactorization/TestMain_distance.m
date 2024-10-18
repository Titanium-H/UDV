%% Save Name
saveName = ['./results/SD',num2str(MC)];
if noisy, saveName = [saveName,'N']; end %#ok
saveName = [saveName,'_distance=',num2str(xi,'%1.0e')];

%% Fix random seed
rng(0,'twister');

%% Load data
fprintf('Loading the data\n');
dataName = ['./data/SyntheticData_MC',num2str(MC),'.mat'];
load(dataName);
if noisy
    b = b + randn(size(b))*norm(b)*1e-2;
end

%% Initialization
maxit = 1e6;
U0 = U0/norm(U0,'fro')*xi;
D0 = eye(n);
stepsize = 1e-2; % step-size. try also 0.0001, 0.001, 0.01, 0.1, 1

%% Solve

% with UU' factorization
fprintf('Solving with UU\n');
[U_uu, err_uu] = UU_factorization(A,AT,b,U0,stepsize,maxit);
X_uu = U_uu*U_uu'; X_uu = 0.5*(X_uu+X_uu'); sval_uu = svd(X_uu);

% with UDU' factorization
fprintf('Solving with UDU\n');
[U_udu, D_udu, err_udu] = UDU_factorization(A,AT,b,U0,D0,stepsize,maxit);
X_udu = U_udu*D_udu*U_udu'; X_udu = 0.5*(X_udu+X_udu'); sval_udu = svd(X_udu);

%% Save results
fprintf('Saving the results\n');
if ~exist('results','dir'), mkdir('dir'); end
save(saveName,'-v7.3');
