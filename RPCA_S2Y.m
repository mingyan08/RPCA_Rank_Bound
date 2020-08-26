function [L, S, out] = RPCA_S2Y(D, prox_S, prox_L, t, para)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   Algorithm 1 without acceleration for RPCA
%  
%   Minimize  0.5* ||L + S -D||^2 + f(S) + g(L)
%
%   Input:   
%           t:  stepsize
%           para.tol: 1e-4
%           para.MAX_ITER : maximum number of iterations
%           para.fcnvalue : the operator to compute the function value
%   Output:
%           out.errL: relative error for each iteration
%           out.fcnvalue: function value for each iteration
%           out.time: time for each iteation
%           out.i: the total number of iterations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    L           = zeros(size(D));   

    tol         = para.tol;
    errL        = zeros(para.MAX_ITER, 1);
    fcnvalue    = zeros(para.MAX_ITER, 1);
    time        = zeros(para.MAX_ITER, 1);
    S           = prox_S(D - L);   % the gradient descent step
    timerval    = tic;
    for i = 1: para.MAX_ITER
        L_new   = prox_L(L - t * (S + L - D), t);
        S       = prox_S(D - L_new);   % the gradient descent step
        errL(i)  = norm(L_new - L,'fro');
        time(i)  = toc(timerval);
        fcnvalue(i) = para.fcnvalue(L_new , S, D);
        timerval = tic;
        L   = L_new;
        if errL(i)/norm(L, 'fro') < tol 
            break
        end
        %[i, errL(i)]
    end
    out.errL        = errL;
    out.fcnvalue    = fcnvalue;
    out.time        = time;
    out.i = i;
end

