function [err, W_out, Z_out] = Refine_EM_gen(c,Z, y,X,  K) %round to K

[N,D] = size(X);

[c2, ind] = sort(c,'descend');
Z1 = Z(:,ind(1:min(K,end)));


%solve with EM (gen mix) initialized by W1
EM_T=100;
eps = 1e-5;
tau = 1e-4;
verbose=0;
[err,W_out,Z_out]=EM_gen( min(K,length(c)), EM_T, eps, tau, verbose, X, y, Z1);

