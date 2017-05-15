function [X,Phi,y,Z0,W0] = genData()

seed = 2;
rand('seed',seed);
randn('seed',seed);
N = 600;
P = 10;
K = 3;

W0 = rand(K,1+P)*2-1; %with bias
%X = rand(N,D)*2-1;
X = rand(N,1)*2-1;
Phi = polyExp(X,P);

Z0 = randMul(N,K);
y = sum(Z0 .* (Phi*W0'), 2);

%plot(X,y,'x');
%saveas(gcf, '~/public_html/figures/tmp.pdf', 'pdf');
