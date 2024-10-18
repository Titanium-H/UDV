function [U,D,err] = UDU_factorization(A,AT,b,U0,D0,ss,maxit)
%UDU_FACTORIZATION_PSD for least squares problem
%   Inputs: 
%   - A: handle for the linear measurement map
%   - AT: handle for the adjoint of the measurement map
%   - b: observations
%   - U0: initial estimate for U
%   - D0: initial estimate for D
%   - ss: stepsize
%   - maxit: number of iterations
%   Outputs:
%   - U: factor U
%   - D: factor D
%   - err: convergence error (least-squares)

n = size(D0,1);
U = U0; D = D0;
err = nan(maxit,1);
X = U*D*U'; X = 0.5*(X+X');
for t = 1:maxit
    
    X = U*D*U'; X = 0.5*(X+X');
    AXb = A(X) - b;
    err(t) = norm(AXb)^2;
    AtXb = AT(AXb);
    gradU = (AtXb + AtXb')*U*D;
    gradD = U'*AtXb*U; % also try non-simultaneous updates    

    U = U - ss*gradU;
    nUfro = norm(U,'fro');
    if nUfro > 1
        U = U./nUfro;
    end

    D = diag(max(diag(D) - ss*diag(gradD),0));
end

end

