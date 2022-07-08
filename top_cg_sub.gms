$title Team orienteering in a depot network
$if not set gdxfile $set gdxfile _gams_py_gdb0.gdx
set i;
alias(i,j,i1,j1,j2,i1,i2);

parameter
    c(i,j), v(i), B, p, pathcost
    rb(i) reduced benefit = v(i) - master_once.m(i)
    msm(j) master_start.m(j);
$gdxin %gdxfile%
$load i,c,B,p,v,pathcost
$gdxin
*display i,c,B,p,v,pathcost;

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
* Initial value
rb(i)$(ord(i) > p) = v(i);
msm(j)$(ord(j) <= p) = 0;
