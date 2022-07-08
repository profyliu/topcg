$title Team orienteering in a depot network
$ontext
Solve the set covering LP using model instance in python
$offtext
$if not set gdxfile $set gdxfile _gams_py_gdb0.gdx
$if not set maxpaths $set maxpaths 200
set i;
alias(i,j);

parameter p number of depots;
parameter wj(j) number of drones initially located at depot j;

set h set of paths /h0*h%maxpaths%/;
parameters
    hs(h,j) 1 if path h starts from depot j
    hi(h,i) 1 if path h visits order i
    vh(h) value of path h
;

$gdxin %gdxfile%
$load i,p,wj
$gdxin

positive variable select(h) 1 if path h is selected;
variable master_objval;
equation master_obj, master_start, master_once;
master_obj..
    master_objval =e= sum(h$(ord(h)>1), vh(h)*select(h));
master_start(j)$(ord(j) <= p)..
    sum(h$(ord(h)>1), hs(h,j)*select(h)) =l= wj(j);
master_once(i)$(ord(i) > p)..
    sum(h$(ord(h)>1), hi(h,i)*select(h)) =l= 1;
model master /master_obj, master_start, master_once/;
hs('h0','depot0') = 1;
hi('h0','0') = 1;
vh('h0') = 0;
