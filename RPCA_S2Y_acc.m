function [L, S, out] = RPCA_S2Y_acc(D, prox_S, prox_L, t, para)


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

    L           = zeros(size(D));   % This only consider the matrix D, when D is not a matrix, ????
    S           = L;
    tol         = para.tol;
    errL        = zeros(para.MAX_ITER, 1);
    fcnvalue    = zeros(para.MAX_ITER, 1);
    time        = zeros(para.MAX_ITER, 1);
    timerval    = tic;
    t_old       = 0;
    t_new       = 1;
    L_old       = L;
    L_new       = L;
    Z           = L;
    c           = para.fcnvalue(L_old, S, D);
    delta       = 1;
    eta         = 0.6;
    q       = 1;
    for i = 1: para.MAX_ITER
        %L_new   = prox_L(L - t * (S + L - D), t);
        %t_new   = (sqrt(1 + 4 * t_old^2) + 1)/2;
        L       = L_new + t_old/t_new * (Z-L_new) + (t_old - 1)/t_new * (L_new-L_old);
        S       = prox_S(D - L);
        Z_new   = prox_L(L - t * (S + L - D), t);
        
        %L_new   = prox_L(L - t * (S + L - D), t);
           % the gradient descent step
        
        fcnZ    = para.fcnvalue(Z_new, prox_S(D - Z_new), D);
        if i == 1
            fcnvalue(i) = fcnZ;
        end
        
        if fcnZ <= c - delta * norm(Z_new - L, 'fro')^2
            L_hat = Z_new;
        else
            S = prox_S(D - L_new);
            v = prox_L(L_new - t * (S + L_new - D), t);
            fcnV  = para.fcnvalue(v, prox_S(D - v), D);
            if fcnZ <= fcnV
                L_hat = Z_new;
            else
                L_hat = v;
            end
        end
        if i>1
            fcnvalue(i) = fcnX;
        end
        fcnX    = para.fcnvalue(L_hat, prox_S(D - L_hat), D);
        errL(i)  = norm(L_new - L_old,'fro');
        time(i)  = toc(timerval);
%        fcnvalue(i) = para.fcnvalue(L_new, prox_S(D - L_new), D);
        if errL(i)/norm(L_old, 'fro') < tol 
            break
        end
        L_old   = L_new; 
        L_new   = L_hat;
        Z       = Z_new;
        t_old   = t_new;
        t_new   = (sqrt(1 + 4 * t_new^2) + 1)/2;
        q   = eta * q + 1;
        c   = (eta * q * c + fcnX)/q;
%        [i, errL(i)]
    end
    L = L_new;
    out.errL        = errL;
    out.fcnvalue    = fcnvalue;
    out.time        = time;
    out.i = i;
end


