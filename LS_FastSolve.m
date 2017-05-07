function [w,a] = LS_FastSolve(c, ETE, ETy, y, E, Tau )

[N,DK] = size(E);
K = length(c);
D = floor(DK/K);

ce = reshape( ones(D,1)*c', [DK,1] );

S=find(ce~=0);
Sb=find(ce==0);

w = zeros(DK,1);
w(Sb)=0;
w(S) =  ( ETE(S,S)+diag(Tau./ce(S)) ) \ ETy(S);

%w = Q*( inv( diag(S_diag)+spdiags(Tau./c_expand,0,DK,DK) ) )*QTETy;
a = (y-E*w);
