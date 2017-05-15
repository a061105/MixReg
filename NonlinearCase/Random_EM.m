function [alpha,Z,pred_funcs,best_rmse] = Random_EM(y,X,ker_func,T_EM,T_random, K, Z_init)

[N,D] = size(X);

'compute kernel matrix...'
Q = kernelMat(ker_func, X);

best_A=-1;
best_rmse = 1e300;
for  t = 1:T_random

	if isempty(Z_init)
			Z1 = randMul(N,K);
	else
			Z1 = Z_init;
	end

	[alpha2,Z2,A,rmse] = EM(y, Z1, [], Q, T_EM);
	if rmse < best_rmse
			best_rmse = rmse;
			alpha = alpha2;
			Z = Z2;
			best_A = A;
	end
	['EM_retrial=' num2str(t) ', best_rmse=' num2str(best_rmse)]

end

Xcell = num2cell(X,2);

pred_funcs = cell(K,1);
for k = 1:K
		pred_funcs{k} = @(x) best_A(:,k)' * cellfun(@(xi) ker_func(xi,x), Xcell);
end
