%[X,Phi,y,Z0,W0] = genData();
[y,data] = libsvmread('../../../data/triazines_scale');

%Construct Label
J = 3;
%Z0 =  full(data(:,J));
z = data(:,J);
N = length(z);
labels = unique(z);
K0 = length(labels);
Z0 = zeros(N,K0);
for i =1:N
		Z0(i,:) = (z(i)==labels');
end
[N,K0] = size(Z0)

gamma = 0.001;
ker_func = @(x1,x2)rbf_kernel(x1,x2,gamma);
%Choose Response
%best_ratio = -1e300;
%best_J = -1;
%for J = 1:size(data,2)
		X = data(:,setdiff(1:size(data,2),J));
		%y = data(:,J);
		
		[N,D] = size(X);
		
		[alpha,Z,pred_funcs,rmse] = Random_EM(y,X, ker_func, 1, 1, K0, Z0);
		rmse

		Q = kernelMat(ker_func, X);
		alpha = (Q + eye(N)) \ y;
		y2 = Q*alpha;
		rmse2 = norm(y-y2)/sqrt(N);
		rmse2

%		ratio = rmse2 / rmse;
%		if ratio > best_ratio
%				best_ratio = ratio;
%				best_J = J;
%
%				best_ratio
%		end
%end

%best_ratio
%best_J
