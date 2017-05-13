function [X,y,Z0,W0] = genData2()

%load('../Sampledata/data1_k=3/data.mat');
%Z0 = Z;
%W0 = W;
%y = Y;
%X = X;
seed = 4;
rand('seed',seed);
randn('seed',seed);
N = 10000;
D = 20;
K = 5;
%X = rand(N,D)*2-1;
X = randn(N,D);
%V0 = randn(K,D);
W0 = randn(K,D);
Z0 = binornd(1,0.5,[N,K]);
%Z0 = zeros(N,K);
%X = zeros(N,D);
%for i = 1:N
%		k = ceil(rand*K);
%		Z0(i,k) = 1;
		%X(i,:) = V0(k,:)+randn(1,D)*0.1;
%end
y = sum(Z0 .* (X*W0'), 2);
