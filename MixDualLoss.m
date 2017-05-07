function  dobj  =  MixDualLoss(y,X,Zc,a, Tau)

[N,D] = size(X);

Wtmp = Zc'* spdiags(a,0,N,N) *X; %K by D

dobj = -norm(Wtmp,'fro')^2/(2*Tau) + y'*a  - norm(a)^2/2;
