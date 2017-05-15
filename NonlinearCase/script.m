[X,Phi,y,Z0,W0] = genData();

[N,D] = size(X);
[N,K0] = size(Z0);

%gamma = 1;
%ker_func = @(x1,x2)rbf_kernel(x1,x2,gamma);

%ker_func = @(x1,x2) x1*x2';

deg=10;
b=1;
a=1;
ker_func = @(x1,x2)poly_kernel(x1,x2,a,b,deg);

T_random=100;
T_EM = 100;
%[alpha,Z,pred_funcs,rmse] = Random_EM(y,X, ker_func, T_EM, T_random, K0, []);
%[alpha,Z,pred_funcs,rmse] = Random_EM(y,X, ker_func, 1, 1, K0, Z0);
%rmse

Lambda = N*5;
KernelMixLasso(y, X, Lambda, ker_func, Z0, 'Weichen');

plot(X,y,'o','color',[0.3,0.3,0.3]);
hold on;
x=-1:0.01:1;
for k = 1:K0
		plot(x, arrayfun(pred_funcs{k},x),'LineWidth',2);
		hold on;
end
hold off;
saveas(gcf,'~/public_html/figures/tmp.pdf','pdf');
