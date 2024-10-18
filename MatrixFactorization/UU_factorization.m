function [U,err] = UU_factorization(A,AT,b,U0,ss,maxit)
%UU_FACTORIZATION_PSD for least squares problem
%   Inputs: 
%   - A: handle for the linear measurement map
%   - AT: handle for the adjoint of the measurement map
%   - b: observations
%   - U0: initial estimate for U
%   - ss: stepsize
%   - maxit: number of iterations
%   Outputs:
%   - U: factor U
%   - err: convergence error (least-squares)

U = U0;
err = nan(maxit,1);

for t = 1:maxit
    U_old = U;
    X = U*U';
    AXb = A(X) - b;
    err(t) = norm(AXb)^2;
    AtXb = AT(AXb);
    grad = (AtXb+AtXb')*U;
    U = U - ss*grad; 
    
    if norm(U-U_old,'fro') < 1e-12
        err(t+1:end) = [];
        break;
    end

end

end

