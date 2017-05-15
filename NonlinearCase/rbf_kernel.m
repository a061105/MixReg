function kval = rbf_kernel(x,y, gamma)

tmp = x-y;

kval = exp(-gamma*sum(tmp.*tmp));
