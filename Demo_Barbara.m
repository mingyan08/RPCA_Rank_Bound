%% load the image

clear all 
close all
global lambda mu p M 

rng(100,'twister');
X       = double(imread('figures\barbara.png'));
[m,n]   = size(X);
X_0     = X;

%---salt-and-pepper noise-----
c_ratio         = 0.2; % corruption ratio
c_num           = round(c_ratio * m * n);
noise           = randn(1, c_num);
noise(noise>=0) = 255;
noise(noise<0)  = 0;
J               = randperm(m*n, c_num); 
X(J)            = noise;

%---small noise-----
D               = X + 4 * randn(size(X));

%  parameters
lambda      = 0.05;
mu          = 1;
p           = 50;
t           = 1;

[U, S, V]   = svds(D, p);
M           = U * S;
M_0         = M;


gradient    = @(D)   feval(@prox_S, D);
proximal    = @(D, t) feval(@prox_L, D, t);
proximal2   = @(D, t) feval(@prox_L_GN, D, t);
fcnvalue    = @(L, S, D) feval(@objective, L, S, D);
para.MAX_ITER   = 1e4;
para.fcnvalue   = fcnvalue;
para.tol        = 1e-4;

corrupted = figure(1); 
imshow(uint8(D));
myprint('figures\corrupted_barbara', corrupted);
fprintf('Noisy: RE %f; PSNR %f\n', norm(D - X_0, 'fro')/norm(X_0, 'fro'), psnr(X_0, D, max(X_0(:)))) 
%% Parameter selection 
%  mu = 0; lambda = 0.03
%  mu = 0.5; lambda = 0.06       
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

mu      = 0;
lambda  = 0.03;
M       = M_0;
t       = 1;
[L_Shen, S_Shen,  out_Shen]    = RPCA_S2Y(D, gradient, proximal2, t, para);


recovered_shen = figure(2);
imshow(uint8(L_Shen));
myprint('figures\recovered_barbara_shen', recovered_shen);
fprintf('Shen et al.: RE %f; PSNR %f\n', norm(L_Shen - X_0, 'fro')/norm(X_0, 'fro'), psnr(X_0, L_Shen, max(X_0(:)))) 

mu      = 0.5;
lambda  = 0.06;
M       = M_0;
t       = 1.7;
[L_acc, S_acc,  out_acc]    = RPCA_S2Y_acc(D, gradient, proximal2, t, para);


recovered = figure(3);
imshow(uint8(L_acc));
myprint('figures\recovered_barbara', recovered);
fprintf('Proposed: RE %f; PSNR %f\n', norm(L_acc - X_0, 'fro')/norm(X_0, 'fro'), psnr(X_0, L_acc, max(X_0(:)))) 


%% Shen et al.'s acceleration
mu      = 0;
lambda  = 0.03;
M       = M_0;
t       = 1;
[L_Shen_acc, S_Shen_acc,  out_Shen_acc]    = RPCA_S2Y_acc(D, gradient, proximal2, t, para);

recovered_shen = figure(20);
imshow(uint8(L_Shen_acc));
myprint('figures\recovered_barbara_shen_acc', recovered_shen);
fprintf('Shen et al.: RE %f; PSNR %f\n', norm(L_Shen_acc - X_0, 'fro')/norm(X_0, 'fro'), psnr(X_0, L_Shen_acc, max(X_0(:)))) 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  RE 0.144668; PSNR 22.367733
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Compare the algorithms performance
f_min   = 5.907781362225356e+05;

mu      = 0.5;
lambda  = 0.06;
M       = M_0;
t       = 1.7;
[L_SVD, S_SVD,  out_SVD]    = RPCA_S2Y(D, gradient, proximal, t, para);

M       = M_0;
[L_Alg, S_Alg,  out_Alg]    = RPCA_S2Y(D, gradient, proximal2, t, para);

fcn_compare = figure(4);
semilogy(cumsum(out_SVD.time), out_SVD.fcnvalue - f_min,'Color',[0.4660, 0.6740, 0.1880],'Linewidth',2); hold on
semilogy(cumsum(out_Alg.time), out_Alg.fcnvalue - f_min, ':','Linewidth', 2);
semilogy(out_acc.time, out_acc.fcnvalue - f_min,'--','Linewidth',2);
h1 = legend('general SVD', 'Alg. 1', 'Alg. 2');
ylabel('f^k - f^*');xlabel('time (s)');
ylim([1e4,1.5e6])
set(gca,'FontSize',14)
myprint('figures\fcn_compare_barbara', fcn_compare);

fprintf('Iterations: Non_acc: %d; Acc: %d\n', out_Alg.i, out_acc.i) 


L_compare = figure(5);
semilogy(cumsum(out_SVD.time), out_SVD.errL, 'Color',[0.4660, 0.6740, 0.1880], 'Linewidth',2); hold on
semilogy(cumsum(out_Alg.time), out_Alg.errL, '--','Linewidth',2);
semilogy(out_acc.time, out_acc.errL, ':','Linewidth', 2);
hl = legend('general SVD','Alg. 1','Alg. 2');
ylabel('relative error');xlabel('time (s)');
set(gca,'FontSize',14)
myprint('figures\L_compare_barbara', L_compare);


%%
%%%%%%%%%%%%%%%%%
%  Functions    %
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
    global mu M
    Afun        = @(X)D*(D'*X); 
    opt2.tol    = 1e-8;
    opt2.maxit  = 1000;
    [M, Out]    = GN_slrp(Afun, M, opt2); 
    [U, S, V]   = svd(M, 'eco');
    S           = diag(max(diag(S) - (t * mu), 0));
    L           = (U * S * V') * (Out.Xi' * D);
end