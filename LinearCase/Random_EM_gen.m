function [best_err,Wout,Zout] = Random_EM(y,X, K, T_random,T_EM)

[N,D] = size(X);

eps = 1e-5;
tau = 1e-4;
verbose=0;
best_err = 1e300;
for t = 1:T_random
		W1 = randn(K,D);
		%[W2,Z2] = EM(y,X,W1,T_EM);
	  [err,W2,Z2]=EM_gen_2( K, T_EM, eps, tau, verbose, X, y, W1');	
		W2 = W2';
		err = norm(y-sum(Z2.*(X*W2'),2))/sqrt(N);
		if err < best_err
						best_err = err;
						Wout = W2;
						Zout = Z2;
		end
		['random_trial=' num2str(t) ', best_err=' num2str(best_err)]
end
