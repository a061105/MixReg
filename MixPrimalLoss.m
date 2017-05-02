function pobj = MixPrimalLoss(y, E, w, Tau )

y2 = E*w;

pobj = norm(y-y2)^2/2 + Tau*norm(w)^2/2;
