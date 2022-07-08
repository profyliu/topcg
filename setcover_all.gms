$title Team orienteering in a depot network
$if not set gdxfile $set gdxfile _gams_py_gdb0.gdx
set i,k;
alias(i,j);
set h set of paths;
set hs(h,j) starting depot of path
set he(h,j) ending depot of path
set hi(h,i) path order incidence;
parameter p, vh(h), pathcost;
parameter w(i,k) 1 if drone k is initially located at depot i;
$gdxin %gdxfile%
$load i,k,h,hs,he,hi,p,vh,w,pathcost
$gdxin
*display i,k,h,hs,he,hi,p,vh,w,pathcost;

binary variable
    y(h) 1 if path h is included
variable
    objval objective value
;
equations eq_obj, eq_cap, eq_once;

eq_obj..
    objval =e= sum(h,vh(h)*y(h)) - pathcost*sum(h, y(h));
eq_cap(j)$(ord(j) <= p)..
    sum(h$hs(h,j), y(h)) =l= sum(k,w(j,k));
eq_once(i)$(ord(i) > p)..
    sum(h$hi(h,i), y(h)) =l= 1;
model top /eq_obj, eq_cap, eq_once/;
option optcr=0,limrow=0,limcol=0;
top.solprint=no;
solve top using mip maximize objval;
parameter n_path_used, n_orders_served, ms, ub, gap;
n_path_used = sum(h,y.l(h));
n_orders_served = sum(i, sum(h$hi(h,i), y.l(h)));
ms = top.modelstat;
ub = top.objest;
gap = 2;
gap$(objval.l >1e-3) = (ub - objval.l)/objval.l;

