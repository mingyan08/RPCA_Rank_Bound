%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  Code to reproduce the synthetic TABLE 2 from the paper.
%  You may need to change paramters to get all result.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all 
close all

global lambda mu p M mask

rng(100,'twister');
m           = 500;
n           = 500;
r_ratio     = 0.05; % rank ratio
k           = round(r_ratio * m); 
A           = randn(m,k) * randn(k,n);
c_ratio     = 0.05; % corruption ratio
J           = randperm(m * n); 
J           = J(1:round(c_ratio * m * n));    
y           = A(:);
mean_y      = mean(abs(y));
noise       = 6 * mean_y * rand(1, round(c_ratio * m * n)) - 3 * mean_y;
y(J)        = noise;

%---small noise-----
sigma       = 0.05;
if c_ratio == 0.05
    sigma = 0.01;
end
small_noise = sigma * randn(size(y));
y           = y + small_noise;

D 		= reshape(y, m, n);
mask 	= rand(m,n) < 0.5;
D 		= mask.*D;

p 		= k + 5;

[U, S, V]   = svds(D, p);
M  			= U * S;
M_0 		= M;

gradient    = @(D)   feval(@prox_S, D);
proximal    = @(D, t) feval(@prox_L, D, t);
proximal2   = @(D, t) feval(@prox_L2, D, t);
fcnvalue    = @(L, S, D) feval(@objective, L, S, D);
para.MAX_ITER 		= 500;
para.fcnvalue 		= fcnvalue;
para.accelerated 	= 1;
para.tol 			= 1e-4;

%%  TABLE 2: The results for TABLE 2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  The TABLE 2 in the paper
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
%%% This numerical experiment solves the matrix completion problem.
%
% We change the number of missing entries from the matrix
% 
% Case #1 c_ratio = 0.2 sigma = 0.05  missing 10%  
%         mu = 0.5, lambda = 0.04
% Case #2 c_ratio = 0.2 sigma = 0.05  missing 20%  
%         mu = 0.5, lambda = 0.04
% Case #3 c_ratio = 0.2 sigma = 0.05  missing 50%  
%         mu = 0.5, lambda = 0.04
% Case #4 c_ratio = 0.05 sigma = 0.01  missing 50%  
%         mu = 0.1, lambda = 0.01
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if c_ratio == 0.2
    mu = 0.5;
    lambda = 0.04;
elseif c_ratio == 0.05
    mu = 0.1;
    lambda = 0.01;
end
t 		= 1.7;
M 				= M_0;
[L, S,  out]    = RPCA_completion_acc(D, gradient, proximal2, t, mask, para);
fro_L       	= norm(L - A, 'fro')/norm(A, 'fro');
iter        	= out.i;
    




%%%%%%%%%%%%%%%%%
%%  Functions  %%
%%%%%%%%%%%%%%%%%
function val = objective(L, S, D)
    global mu lambda mask
    val = 0.5 * norm(mask.*(L + S - D),'fro')^2 + lambda * sum(sum(abs(mask.*S))) + mu * norm(svd(L),1); 
end

function S = prox_S(D) 
    global lambda mask
    S = sign(D) .* max(abs(D) - lambda, 0);
    S = mask .* S;
end

function L = prox_L(D, t)  % with large SVD
    global mu p 
    [U, S, V]   = svds(D, p);
    sd          = diag(S);
    sd          = max(sd - (t * mu), 0);
    S           = diag(sd);
    L           = U * S * V';
end

function L = prox_L2(D, t)   % without large SVD
    global mu M
    Afun = @(X)D*(D'*X); 
    opt2.tol    = 1e-8;
    opt2.maxit  = 1000;
    [M, Out]    = GN_slrp(Afun, M, opt2); 
    if Out.iter > 520
        Out.iter
    end
    [U, S, V]   = svd(M, 'eco');
    S          = diag(max(diag(S) - (t * mu), 0));
    L           = (U * S * V') * (Out.Xi' * D);
end