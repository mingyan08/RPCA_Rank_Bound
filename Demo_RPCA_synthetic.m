%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  Code to reproduce the synthetic results from the paper.  
%   Run by sections. DONOT run all!!
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  The initialization for all the codes
clear all 
close all

global lambda mu p M inneriter

rng(100,'twister');
m           = 500;
n           = 500;
r_ratio     = 0.05; % rank ratio
k           = round(r_ratio * m); 
A           = randn(m,k) * randn(k,n);
c_ratio     = 0.20; % corruption ratio
J           = randperm(m * n); 
J           = J(1:round(c_ratio * m * n));    
y           = A(:);
mean_y      = mean(abs(y));
noise       = 6 * mean_y * rand(1, round(c_ratio * m * n)) - 3 * mean_y;
y(J)        = noise;

%---small noise-----
small_noise = 0.05 * randn(size(y));
y           = y + small_noise;

D           = reshape(y, m, n);

p           = k + 5 ;
t           = 1.7;
[U, S, V]   = svds(D, p);
M  = U * S;
M_0 = M;

gradient    = @(D)   feval(@prox_S, D);
proximal    = @(D, t) feval(@prox_L, D, t);
proximal2   = @(D, t) feval(@prox_L_GN, D, t);
fcnvalue    = @(L, S, D) feval(@objective, L, S, D);
para.MAX_ITER = 1000;
para.fcnvalue = fcnvalue;
para.accelerated = 1;
para.tol = 1e-4;

%% Parameter selection 
% Case #1:  r = 25 and s = 20.  
%           mus = [0: 0.1: 2]
%           lambdas = [0.01:0.01:0.2]
%           optimal for Shen. lambda = 0.01
%           optimal for propsoed  mu = 0.6, lambda =0.04
% 
% Case #2: r = 50 and s = 20.
%           mus = [0: 0.1: 1]
%           lambdas = [0.01:0.01:0.1]
%           optimal for Shen.    lambda = 0.01
%           optimal for propsoed  mu = 0.5, lambda =0.04
% 
% Case #3: r = 25 and s = 40
%           mus = [0: 0.1: 1]
%           lambdas = [0.01:0.01:0.1]
%           optimal for Shen.    lambda = 0.01
%           optimal for propsoed  mu = 0.3, lambda =0.03          
%
%%  Compute the contour map data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  The FIGURE 1 in the paper
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mus     = 0: 0.1: 2;
lambdas = 0.01:0.01:0.2;
fro_L   = zeros(length(mus)*length(lambdas), 1);
iter    = zeros(length(mus)*length(lambdas), 1);
t       = 1.7;
ind     = 1;
para.MAX_ITER   = 400;
for mu = mus
    mu
    for lambda = lambdas
        tic
        M = M_0;
        [L, S,  out]        = RPCA_S2Y_acc(D, gradient, proximal2, t, para);
        fro_L(ind) = norm(L - A, 'fro')/norm(A, 'fro');
        iter(ind) = out.i;
        ind = ind + 1;
        toc
    end
end
fro_all     = reshape(fro_L, length(lambdas),length(mus));
iter_all    = reshape(iter, length(lambdas),length(mus));

[X,Y]       = meshgrid(mus,lambdas);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Alternatively, load the data from contour.mat
%  load('contour.mat')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure(2)
[C,h] = contourf(X,Y,fro_all,[0.007,0.008,0.01,0.02,0.04,0.08]);
clabel(C,h,'FontSize',15,'Color','red')
ylabel('\lambda','fontsize',20);xlabel('\mu','fontsize',20);
set(gca,'FontSize',14)
myprint('figures\contour')

%%  TABLE 1: The results for TABLE 1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  The TABLE 1 in the paper
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

para.MAX_ITER = 5000;

mu      = 0.;
lambda  = 0.02;
M       = M_0;
t       = 1;
[L, S,  out]    = RPCA_S2Y(D, gradient, proximal2, t, para);
fro_L(1)        = norm(L - A, 'fro')/norm(A, 'fro');
iter(1)         = out.i;

mu      = 0.6;
lambda  = 0.04;
M       = M_0;
t       = 1.7;
[L, S,  out]    = RPCA_S2Y(D, gradient, proximal2, t, para);
fro_L(2)        = norm(L - A, 'fro')/norm(A, 'fro');
iter(2)         = out.i;
objective(L, S, D)


mu      = 0.6;
lambda  = 0.04;
M       = M_0;
t       = 1.7;
[L, S,  out]    = RPCA_S2Y_acc(D, gradient, proximal2, t, para);
fro_L(3)        = norm(L - A, 'fro')/norm(A, 'fro');
iter(3)         = out.i;
objective(L, S, D)


%%  Figure 2: Change p values
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  The FIGURE 2 in the paper
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
para.MAX_ITER = 5000;
for i = 1:21
    p           = k - 11 + i;
    [U, S, V]   = svds(D, p);
    M           = U * S;
    M_0         = M;

    mu          = 0.6;
    lambda      = 0.04;
    t           = 1.7;
    [L, S,  out]    = RPCA_S2Y_acc(D, gradient, proximal2, t, para);
    fro_L_new(i)    = norm(L - A, 'fro')/norm(A, 'fro')

    mu      = 0;
    lambda  = 0.02;
    t       = 1.0;
    M       = M_0;
    [L, S,  out]    = RPCA_S2Y(D, gradient, proximal2, t, para);
    fro_L_Shen(i)   = norm(L - A, 'fro')/norm(A, 'fro')
end

figure(3)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Alternatively, load the data from contour.mat
%  load('contour.mat')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
plot([k-10:1:k+10], fro_L_Shen, 'r*-')
hold on
plot([k-10:1:k+10], fro_L_new, 'bo-')
legend('Shen et al.', 'Alg. 2')
ylabel('relative error of L','fontsize',20);xlabel('rank (p)','fontsize',20);
set(gca,'FontSize',14)
myprint('figures\robust')

%%  Additional: Change Stepsize
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  t has to be smaller than 2, and 1.7 is a good choice. 
%  (Consider r_ratio = 0.1 as well)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
para.MAX_ITER = 500;

p           = k + 5;
[U, S, V]   = svds(D, p);
M           = U * S;
M_0         = M;
mu          = 0.6;
lambda      = 0.04;
ind         = 1;
for t = 0.5:0.1:2.0
    M = M_0;
    [L, S,  out]        = RPCA_S2Y_acc(D, gradient, proximal2, t, para);
    fro_L_stepsize(ind) = norm(L - A, 'fro')/norm(A, 'fro');
    iter_stepsize(ind)  = out.i;
    ind = ind + 1;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  The final solution are the same. The number of iterations: 
%                          164   147   132   124   111   103   
%   97    89    86    84    77    75    68    60    59    63
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Additional: Change Initial M
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  The initial M does not matter. A random matrix is fine. 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

para.MAX_ITER = 500;

p           = k + 5;
[U, S, V]   = svds(D, p);
M           = U * S;
M_0         = M;
mu          = 0.6;
lambda      = 0.04;
t           = 1.7;

for ind = 1:10
    ind
    inneriter   = 0;    
    M           = rand(m, p);
    if ind == 1
        M = M_0;
    end
    [L, S,  out]    = RPCA_S2Y_acc(D, gradient, proximal2, t, para);
    fro_L_init(ind) = norm(L - A, 'fro')/norm(A, 'fro');
    iter_init(ind)  = out.i;
    inner_iter(ind) = inneriter;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  The iterations are the same except the first G-N requires different 
%  number of inner iterations. 
% 
%  Total number of inner iterations for G-N:
%   480   523   510   463   409   443   515   436   554   459
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%
%%  Functions  %%
%%%%%%%%%%%%%%%%%
function val = objective(L, S, D)
    global mu lambda 
    val = 0.5 * norm(L + S - D,'fro')^2 + lambda * sum(sum(abs(S))) + mu * norm(svd(L),1); 
end

function S = prox_S(D) 
    global lambda 
    S = sign(D) .* max(abs(D) - lambda, 0);
end

function L = prox_L(D, t)  % with large SVD
    global mu p 
    [U, S, V]   = svds(D, p);
    S           = diag(max(diag(S) - (t * mu), 0));
    L           = U * S * V';
end

function L = prox_L_GN(D, t)   % without large SVD
    global mu M inneriter
    Afun        = @(X)D*(D'*X); 
    opt2.tol    = 1e-8;
    opt2.maxit  = 1000;
    [M, Out]    = GN_slrp(Afun, M, opt2); 
    inneriter   = inneriter + Out.iter;
%    if Out.iter > 50
%        Out.iter
%    end
%    Out.iter
    [U, S, V]   = svd(M, 'eco');
    S           = diag(max(diag(S) - (t * mu), 0));
    temp = diag(S)';
    L           = (U * S * V') * (Out.Xi' * D);
end