function kval = poly_kernel(x,y, a, b, deg)

kval = (a* sum(x.*y) + b)^deg;
