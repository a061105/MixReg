[X,y,Z0,W0] = genData();
[N,D] = size(X);
[N,K] = size(Z0);
%['true obj=' num2str( norm(y-sum(Z0.*(X*W0'),2)).^2 + Tau*norm(W0(:))^2/2 + Lambda*K ) ]

Lambda = N; %Suggested Value = N
Tau = 1;
Gamma = 0;
%[Z,W,c] = MixLassoTest(y,X, Lambda, Tau, Gamma, Z0, W0, [Z0 binornd(1,0.5,[N,100])]);
[Z,W,c] = MixLasso(y,X, Lambda, Tau, Gamma, Z0, W0);
[err,W2,Z2,m] = Refine_EM(c,Z,y,X, K);

m
err
