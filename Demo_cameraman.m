%% load the image

clear all 
close all
global lambda mu p M 

rng(100,'twister');
m = 256;
n = 256;
X = imresize(double(imread('figures\Cameraman.png')),[m, n]);


% ---strictly low-rank-----
[U,S,V]     = svd(X);
s           = diag(S);
s(38:end)   = 0;
X           = U*diag(s)*V';
X_0         = X;

% ---salt-and-pepper noise-----
c_ratio         = 0.2; % corruption ratio
c_num           = round(c_ratio * m * n);
noise           = randn(1, c_num);
noise(noise>=0) = 255;
noise(noise<0)  = 0;
J               = randperm(m*n, c_num); 
X(J)            = noise;

% ---small noise-----
D               = X + 4 * randn(size(X));


%  parameters
k           = round(0.15*m) - 1;
p           = k + 5;

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
myprint('figures\corrupted_cameraman', corrupted);
fprintf('Noisy: RE %f; PSNR %f\n', norm(D - X_0, 'fro')/norm(X_0, 'fro'), psnr(X_0, D, max(X_0(:)))) 

 
%% Parameter selection 
%  mu = 0; lambda = 0.03
%  mu = 0.5; lambda = 0.06       
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mu      = 0;
lambda  = 0.03;
M       = M_0;
t       = 1;
[L_Shen, S_Shen,  out_Shen]    = RPCA_S2Y_acc(D, gradient, proximal2, t, para);


recovered_shen = figure(2);
imshow(uint8(L_Shen));
myprint('figures\recovered_cameraman_shen', recovered_shen);
fprintf('Shen et al.: RE %f; PSNR %f\n', norm(L_Shen - X_0, 'fro')/norm(X_0, 'fro'), psnr(X_0, L_Shen, max(X_0(:)))) 


mu      = 0.5;
lambda  = 0.06;
M       = M_0;
t       = 1.7;
[L_acc, S_acc,  out_acc]    = RPCA_S2Y_acc(D, gradient, proximal2, t, para);


recovered = figure(3);
imshow(uint8(L_acc));
myprint('figures\recovered_cameraman', recovered);
fprintf('Proposed: RE %f; PSNR %f\n', norm(L_acc - X_0, 'fro')/norm(X_0, 'fro'), psnr(X_0, L_acc, max(X_0(:)))) 


%% Compare the algorithms performance
f_min = 1.477266944073798e+05;

[L_SVD, S_SVD,  out_SVD]    = RPCA_S2Y(D, gradient, proximal, t, para);
[L_Alg, S_Alg,  out_Alg]    = RPCA_S2Y(D, gradient, proximal2, t, para);

fcn_compare = figure(4);
semilogy(cumsum(out_SVD.time), out_SVD.fcnvalue - f_min,'Color',[0.4660, 0.6740, 0.1880],'Linewidth',2); hold on
semilogy(cumsum(out_Alg.time), out_Alg.fcnvalue - f_min, ':','Linewidth', 2);
semilogy(out_acc.time, out_acc.fcnvalue - f_min,'--','Linewidth',2);
h1 = legend('general SVD', 'Alg. 1', 'Alg. 2');
ylabel('$f^k - f^\star$','interpreter','latex');xlabel('time (s)');
ylim([2e2,5e5])
set(gca,'FontSize',14)
myprint('figures\fcn_compare_cameraman', fcn_compare);

fprintf('Iterations: Non_acc: %d; Acc: %d\n', out_Alg.i, out_acc.i) 


L_compare = figure(5);
semilogy(cumsum(out_SVD.time), out_SVD.errL, 'Color',[0.4660, 0.6740, 0.1880], 'Linewidth',2); hold on
semilogy(cumsum(out_Alg.time), out_Alg.errL, '--','Linewidth',2);
semilogy(out_acc.time, out_acc.errL, ':','Linewidth', 2);
hl = legend('general SVD','Alg. 1','Alg. 2');
ylabel('relative error');xlabel('time (s)');
set(gca,'FontSize',14)
myprint('figures\L_compare_cameraman', L_compare);




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
    global mu M
    Afun        = @(X)D*(D'*X); 
    opt2.tol    = 1e-8;
    opt2.maxit  = 1000;
    [M, Out]    = GN_slrp(Afun, M, opt2); 
    [U, S, V]   = svd(M, 'eco');
    S           = diag(max(diag(S) - (t * mu), 0));
    L           = (U * S * V') * (Out.Xi' * D);
end