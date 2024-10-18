%% Fix random seed
rng(0,'twister');
close all
%% Problem setting
MC = 1; % change to try different datasets
noisy = 0; % set to 1 to try with inconsistent measurements
maxit = 1e4; % to get quick results // in paper we use 1e6

%% Test setup
xi = 1e-2;          % initial distance to origin
stepsize = 1e-1;   % step-size

%% Load data
fprintf('Loading the data\n');
dataName = ['./data/SyntheticData_MC',num2str(MC),'.mat'];
if ~exist(dataName,'file'), CreateSyntheticData; end
load(dataName);
if noisy
    b = b + randn(size(b))*norm(b)*1e-2; % you can try more or less noise by changing here
end

%% Initialization
U0 = U0/norm(U0,'fro')*xi;
D0 = eye(n);

%% Solve
% with UU' factorization
fprintf('Solving with UU\n');
fprintf('Can take some time\n');
[U_uu, err_uu] = UU_factorization(A,AT,b,U0,stepsize,maxit);
X_uu = U_uu*U_uu'; X_uu = 0.5*(X_uu+X_uu'); sval_uu = svd(X_uu);

% with UDU' factorization
fprintf('Solving with UDU\n');
fprintf('Can take some time\n');
[U_udu, D_udu, err_udu] = UDU_factorization(A,AT,b,U0,D0,stepsize,maxit);
X_udu = U_udu*D_udu*U_udu'; X_udu = 0.5*(X_udu+X_udu'); sval_udu = svd(X_udu);

%% Plot results
figure;
loglog(sval_uu)
hold on
loglog(sval_udu)
title('singular value spectrum')
ylabel('singular values')
xlabel('indices')

figure;
if noisy
    % Start CVX
    cvx_begin
    cvx_precision best
    % Define the optimization variable
    variable xcvx(data.n,data.n) symmetric semidefinite
    % Define the objective function (least squares)
    minimize sum_square(xcvx(data.inds) - data.b)
    %     subjectto xcvx >= 0
    cvx_end
    opt = cvx_optval;
else
    opt = 0;
end
loglog(err_uu-opt)
hold on
loglog(err_udu-opt)
title('convergence rate')
ylabel('objective residual')
xlabel('iteration')

