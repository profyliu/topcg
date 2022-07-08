$title Team orienteering in a depot network
$if not set gdxfile $set gdxfile _gams_py_gdb0.gdx
$if not set maxpaths $set maxpaths 200
set i,k;
alias(i,j,i1,j1,j2,i1,i2);

parameter c(i,j), B, p, v(i),pathcost;
parameter w(i,k) 1 if drone k is initially located at depot i;
$gdxin %gdxfile%
$load i,k,c,B,p,v,w,pathcost
$gdxin
*display i,k,c,B,p,v,w,pathcost;

* Define master problem
set h set of paths /h1*h%maxpaths%/;
set hh(h) dynamic path set
    hs(h,j) path h starts from depot j
    he(h,j) path h ends at depot j
    hi(h,i) path h visits order i
;
parameter hlink(h,i,j);
parameter vh(h) value of path h;
parameter wj(j) number of drones at depot j;
wj(j)$(ord(j) <= p) = sum(k,w(j,k));
positive variable select(h) 1 if path h is selected;
variable master_objval;
equation master_obj, master_start, master_once;
master_obj..
    master_objval =e= sum(hh, vh(hh)*select(hh));
master_start(j)$(ord(j) <= p)..
    sum(hh$hs(hh,j), select(hh)) =l= wj(j);
master_once(i)$(ord(i) > p)..
    sum(hh$hi(hh,i), select(hh)) =l= 1;
model master /master_obj, master_start, master_once/;

* Define subproblem
parameter rb(i) reduced benefit = v(i) - master_once.m(i)
         msm(j) master_start.m;
binary variable
    xi(i)       1 if i is visited by the new path
    xij(i,j)    1 if link i->j is in the new path
    zj(j)       1 if the new path starts from depot j
variable sub_objval, t;
equation sub_obj, sub_zj, sub_L, sub_ini, sub_outi, sub_outj, sub_seq;
sub_obj..
    sub_objval =e= sum(i$(ord(i) > p), rb(i)*xi(i)) - sum(j$(ord(j) <= p), msm(j)*zj(j)) - pathcost;
sub_zj..
    sum(j$(ord(j) <= p), zj(j)) =e= 1;
sub_L..
    sum((i1,i2)$(ord(i1) > p and ord(i2) > p and ord(i1) ne ord(i2)), c(i1,i2)*xij(i1,i2))
    + sum((i,j)$(ord(j) <= p and ord(i) > p), c(i,j)*xij(i,j) + c(j,i)*xij(j,i)) =l= B;
sub_ini(i)$(ord(i) > p)..
    sum(j$(ord(j) ne ord(i)), xij(j,i)) =e= xi(i);
sub_outi(i)$(ord(i) > p)..
    sum(j$(ord(j) ne ord(i)), xij(i,j)) =e= xi(i);
sub_outj(j)$(ord(j) <= p)..
    sum(i$(ord(i) > p), xij(j,i)) =e= zj(j);
sub_seq(i,j)$(ord(i) ne ord(j) and ord(i) > p and ord(j) > p)..
    t(j) - t(i) =g= 1 - (card(i) - p + 1)*(1-xij(i,j));
t.lo(i)$(ord(i) > p) = 0;
t.up(i)$(ord(i) > p) = (card(i) - p);
model sub /sub_obj, sub_zj, sub_L, sub_ini, sub_outi, sub_outj, sub_seq/;


Set hlast(h) 'set of the last path';
hlast(h)$(ord(h) = 1) = yes;

* initialize rb and msm
rb(i)$(ord(i) > p) = v(i);
msm(j)$(ord(j) <= p) = 0;

parameter loop_time, final_time, start_time, niter /0/;
parameter master_time /0/, sub_time /0/;
option limrow=0, limcol=0;
master.solprint=no;
sub.solprint=no;
master.solvelink=5;
sub.solvelink=5;
parameter log_subobjval(h), log_masterobjval(h);
loop_time = timeelapsed;
while(niter < card(h),
    niter = niter + 1;
    start_time = timeelapsed;
    solve sub using mip max sub_objval;
    sub_time = sub_time + timeelapsed - start_time;
    break$(sub_objval.l < 0.00001);
    hs(hlast,j)$(zj.l(j) = 1 and ord(j) <= p) = yes;
    hi(hlast,i)$(xi.l(i) = 1 and ord(i) > p) = yes;
    he(hlast,j)$(ord(j) <= p and sum(i$(ord(i) > p), xij.l(i,j)) > 0) = yes;
    vh(hlast) = sum(i$(ord(i) > p), v(i)*xi.l(i)) - pathcost;
    hh(hlast) = yes;
    log_subobjval(hlast) = sub_objval.l;
* Save the links of the added path
    hlink(hlast,i,j)$(xij.l(i,j) = 1) = 1;
    hlast(h) = hlast(h-1);
    start_time = timeelapsed;
    solve master using lp max master_objval;
    master_time = master_time + timeelapsed - start_time;
* update rb and msm
    rb(i)$(ord(i) > p) = v(i) - master_once.m(i);
    msm(j)$(ord(j) <= p) = master_start.m(j);
    log_masterobjval(hlast) = master_objval.l;
);
loop_time = timeelapsed - loop_time;
parameter iter_lim_reached /0/, last_master_objval, npaths;
iter_lim_reached = 1$(card(hh) = card(h));
last_master_objval = master_objval.l;
npaths = card(hh);
*display hh, hs, hi, he, vh, log_subobjval, log_masterobjval, iter_lim_reached, npaths;

binary variable final_select(h) 1 if path h is selected;
variable final_objval;
equation final_obj, final_start, final_once;
final_obj..
    final_objval =e= sum(hh, vh(hh)*final_select(hh));
final_start(j)$(ord(j) <= p)..
    sum(hh$hs(hh,j), final_select(hh)) =l= sum(k,w(j,k));
final_once(i)$(ord(i) > p)..
    sum(hh$hi(hh,i), final_select(hh)) =l= 1;
model final /final_obj, final_start, final_once/;
final_time = timeelapsed;
solve final using mip max final_objval;
final_time = timeelapsed - final_time;
*display master_objval.l, final_objval.l, select.l, final_select.l, vh;
parameter gap, npath_selected, norders_served;
gap = 1;
gap$(master_objval.l>0) = (master_objval.l - final_objval.l)/final_objval.l;
npath_selected = sum(hh, final_select.l(hh));
norders_served = sum(i$(ord(i) > p), final_once.l(i));
*display gap, hlink, loop_time, final_time, master_time, sub_time, niter, npath_selected, norders_served;



