
is_generalized = 1;
if is_generalized 
	[X,y,Z0,W0] = genData2();
else
	[X,y,Z0,W0] = genData();
end

[N,D] = size(X);
[N,K] = size(Z0);
%['true obj=' num2str( norm(y-sum(Z0.*(X*W0'),2)).^2 + Tau*norm(W0(:))^2/2 + Lambda*K ) ]

if is_generalized
		Lambda = 100*N; %Suggested Value = N
else
		Lambda = N;
end
%Lambda = 100;
Tau = 1;
%[Z,W,c] = MixLassoTest(y,X, Lambda, Tau, Gamma, Z0, W0, [Z0 binornd(1,0.5,[N,100])]);
tic;
[Z,W,c, best_rmse] = MixLasso(y,X, Lambda, Tau, Z0, W0, is_generalized);
toc;

best_rmse
