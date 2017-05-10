
%[X,y,Z0,W0] = genData2();
[X,y,Z0,W0] = genData();


[N,D] = size(X);
[N,K] = size(Z0);
%['true obj=' num2str( norm(y-sum(Z0.*(X*W0'),2)).^2 + Tau*norm(W0(:))^2/2 + Lambda*K ) ]

Lambda = 10*N; %Suggested Value = N
%Lambda = 100;
Tau = 1;
is_generalized = 0;
%[Z,W,c] = MixLassoTest(y,X, Lambda, Tau, Gamma, Z0, W0, [Z0 binornd(1,0.5,[N,100])]);
tic;
[Z,W,c, best_rmse] = MixLasso(y,X, Lambda, Tau, Z0, W0, is_generalized);
toc;

best_rmse
