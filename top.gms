$title Team orienteering in a depot network
$if not set gdxfile $set gdxfile _gams_py_gdb0.gdx
set i,k;
alias(i,j);

parameter c(i,j), B, p, v(i),pathcost;
parameter w(i,k) 1 if drone k is initially located at depot i;
$gdxin %gdxfile%
$load i,k,c,B,p,v,w,pathcost
$gdxin
*display i,k,c,B,p,v,w,pathcost;

binary variable
    y(k) 1 if drone k is used
    x(i,j,k) 1 if location i is visited immediately before location j by drone k
    z(i,k) 1 if order is served by drone k;
variable
    t(i) time of order i is visited
    objval objective value
;
equations
    eq_obj,
    eq_cap total travel distance cannot exceed the battery capacity,
    eq_start a used tour must have one depot to order link,
    eq_end a used tour must have one order to depot link,
    eq_in1k a served order must have an in-link from some node (depot or other order),
    eq_out1k a served order must have an out-link to some node (depot or other order)
    eq_in1 no order will ever have more than 1 in-link,
    eq_out1 no order will ever have more than 1 out-link (redundant implied by eq_in1k + eq_out1k + eq_in1),
    eq_seq visit sequence of orders for subtour elimination,
    eq_w the drone can only depart from the depot where it is parked
;
eq_obj..
    objval =e= sum(i$(ord(i) > p), v(i)*sum(k, z(i,k))) - pathcost*sum(k, y(k));
eq_cap(k)..
    sum((i,j)$(ord(i) ne ord(j)), c(i,j)*x(i,j,k)) =l= B*y(k);
eq_start(k,j)$(ord(j) <= p)..
    sum(i$(ord(i) > p), x(j,i,k)) =e= w(j,k)*y(k);
eq_end(k)..
    sum((i,j)$(ord(i) > p and ord(j) <= p), x(i,j,k)) =e= y(k);
eq_in1k(i,k)$(ord(i) > p)..
    sum(j$(ord(i) ne ord(j)), x(j,i,k)) =e= z(i,k);
eq_out1k(i,k)$(ord(i) > p)..
    sum(j$(ord(i) ne ord(j)), x(i,j,k)) =e= z(i,k);
eq_in1(j)$(ord(j) > p)..
    sum((i,k)$(ord(i) ne ord(j)), x(i,j,k)) =l= 1;
eq_out1(i)$(ord(i) > p)..
    sum((j,k)$(ord(i) ne ord(j)), x(i,j,k)) =l= 1;
eq_seq(i,j)$(ord(i) ne ord(j) and ord(i) > p and ord(j) > p)..
    t(j) - t(i) =g= 1 - (card(i)- p + 1)*(1-sum(k,x(i,j,k)));
t.lo(i)$(ord(i) > p) = 0;
t.up(i)$(ord(i) > p) = (card(i) - p);
model top /eq_obj,
eq_cap,
eq_start, eq_end, eq_in1,
*eq_out1,
eq_in1k, eq_out1k, eq_seq/;
parameter soltime;
soltime = timeelapsed;
solve top max objval using mip;
soltime = timeelapsed - soltime;
parameter n_used, n_served, endw(i), UB, gap, modelstat;
n_used = sum(k, y.l(k));
n_served = sum(i$(ord(i) > p), sum(k, z.l(i,k)));
endw(i)$(ord(i) <= p) = sum(k, w(i,k)) + sum(k, sum(j$(ord(i) ne ord(j)), x.l(j,i,k))) - sum(k, sum(j$(ord(i) ne ord(j)), x.l(i,j,k)));
UB = top.objest;
gap = 2;
gap$(objval.l > 1e-3) = (UB - objval.l)/objval.l;
modelstat = top.modelstat;
*display n_used, n_served, endw, UB, gap, soltime, modelstat;
