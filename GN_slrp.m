function [X,Out] = GN_slrp(Afun,X,opts)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   The Gauss-Newton algorithm to solve 
%    
%   maximize \|X*X' - A\|^2_F   
%
%   based on the paper "An efficient Gauss--Newton algorithm for symmetric 
%       low-rank product matrix approximations." by Xin Liu, Zaiwen Wen, and Yin Zhang.
%       SIAM Journal on Optimization 25.3 (2015): 1571-1608.
%
% 
%  Input:   X       % initial X
%           Afun    % the matrix A in operator form
%           opts    % all options
%           tol     % tolerance for the stopping criteria
%           maxit   % the maximum number of iterations
%           quiet   % display the progress or not
%           freq    % the frequency for progress update
%  Output:  X       % the solution
%           Out     % other output
%           Xi      % X/XtX
%           iter    % the number of iterations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if isnumeric(Afun)
    A    = Afun; 
    Afun = @(X) A * X;
end


%%%%%%%%% initialization %%%%%%%%%%
if ~isfield(opts,'tol');     opts.tol = 1e-4;   end
if ~isfield(opts,'maxit');   opts.maxit = 200;  end
if ~isfield(opts,'quiet');   opts.quiet = 1;    end
if ~isfield(opts,'freq');    opts.freq  = 1;    end
tol   = opts.tol;
maxit = opts.maxit;
quiet = opts.quiet;
freq  = opts.freq;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
k = size(X,2);   % k is the number of columns in X
Ik = speye(k);   % identity k x k matrix

if ~quiet; fprintf('GN(  '); end   % show the progress

XtX     = X'*X;                 % 
nrmX    = sqrt(trace(XtX));     % The norm of X.

for iter = 1:maxit  % main loop    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% main computation
    Y = X / XtX;   % Y = X (X'X)^{-1}
%    [R,p] = chol(XtX);
%    if p > 1; X = X(:,1:p-1); end
%    Y = (X / R) / R';
    Z = Afun(Y);   % Z = AY
    X = Z - X*((Y'*Z - Ik)/2);
    XtX = X'*X;
    nrmXp = nrmX;            
    nrmX = sqrt(trace(XtX));    % computer the numer norm of X
%%%%%%%%%%%%%%%%%%%%%%%%%%%    
%% check stopping
    crit = abs(1 - nrmX/nrmXp);  
%    if crit > tol; count = 0;
%    else count = count + 1;
%    end  % if the norms of X at two iterations are close, we keep count                                 

    done = crit < tol || iter == maxit;  % if count is greater than 0, 
    if (~mod(iter, freq) || done) && ~quiet
        fprintf('iter %4i: crit %.2e \n \n',iter,crit);
    end
    if ~quiet 
        fprintf('\b\b%2i',iter);
        if done; fprintf('): '); end
    end
    if done; break; end    
end

Out.Xi   = Y;
Out.iter = iter;
end % solver
