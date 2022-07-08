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

set h set of paths;
parameters
    hs(h,j) 1 if path h starts from depot j
    hi(h,i) 1 if path h visits order i
    vh(h) value of path h
;

$gdxin %gdxfile%
$load h,i,p,wj,hs,hi,vh
$gdxin

binary variable select(h) 1 if path h is selected;
variable final_objval;
equation final_obj, final_start, final_once;
final_obj..
    final_objval =e= sum(h, vh(h)*select(h));
final_start(j)$(ord(j) <= p)..
    sum(h, hs(h,j)*select(h)) =l= wj(j);
final_once(i)$(ord(i) > p)..
    sum(h, hi(h,i)*select(h)) =l= 1;
model final /final_obj, final_start, final_once/;
solve final max final_objval using mip;
parameter selected;
selected = sum(h,select.l(h));   