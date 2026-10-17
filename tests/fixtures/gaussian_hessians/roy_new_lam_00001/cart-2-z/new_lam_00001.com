echo Rungauss
%chk=uniform_lam_00001.chk
%mem=14000MB
%nprocshared=8
# HF/3-21G Freq NoSymm

HF Cartesian Hessian at fixed geometry
 
 0 1
S1
C1   1 bnd2  
C1   2 bnd3     1 ang3  
C1   3 bnd4     2 ang4     1 dih4  
C1   4 bnd5     3 ang5     2 dih5  
N1   2 bnd6     3 ang6     4 dih6  
C1   6 bnd7     2 ang7     3 dih7  
C1   7 bnd8     6 ang8     2 dih8  
C1   8 bnd9     7 ang9     6 dih9  
C1   9 bnd10    8 ang10    7 dih10 
C1  10 bnd11    9 ang11    8 dih11 
C1  11 bnd12   10 ang12    9 dih12 
N1   8 bnd13    7 ang13    6 dih13 
O1  13 bnd14    8 ang14    7 dih14 
C1   5 bnd15    4 ang15    3 dih15 
C1   3 bnd16    2 ang16    6 dih16 
N1  16 bnd17    2 ang17    3 dih17 
O1  13 bnd18    8 ang18    7 dih18 
H1   9 bnd19    8 ang19    7 dih19 
H1  10 bnd20    9 ang20    8 dih20 
H1  11 bnd21   10 ang21    9 dih21 
H1  12 bnd22   11 ang22   10 dih22 
H2   6 bnd23    7 ang23    8 dih23 
H1   4 bnd24    5 ang24   15 dih24 
H1  15 bnd25    5 ang25    4 dih25 
H1  15 bnd26    5 ang26    4 dih26 
H1  15 bnd27    5 ang27    4 dih27 
variables:
bnd2           1.723800
bnd3           1.368000
ang3         110.550800
bnd4           1.428800
ang4         112.984000
dih4          -0.225000
bnd5           1.357200
ang5         112.925100
dih5          -0.660900
bnd6           1.389300
ang6         128.315700
dih6         179.720800
bnd7           1.375400
ang7         124.337900
bnd8           1.408000
ang8         122.877500
bnd9           1.393000
ang9         121.468100
dih9        -179.804000
bnd10          1.373500
ang10        120.637900
dih10          0.287800
bnd11          1.393800
ang11        118.726800
dih11          0.027700
bnd12          1.374800
ang12        121.068900
dih12         -0.182200
bnd13          1.461700
ang13        122.168900
dih13          0.169700
bnd14          1.211300
ang14        118.417900
dih14        174.028600
bnd15          1.495600
ang15        128.097900
dih15       -178.441800
bnd16          1.427800
ang16        122.582400
dih16          0.537600
bnd17          1.148800
ang17        152.634400
dih17       -179.661000
bnd18          1.228400
ang18        118.768500
dih18         -6.284300
bnd19          1.077500
ang19        117.983000
dih19       -179.702200
bnd20          1.078300
ang20        120.327000
dih20        179.952900
bnd21          1.080200
ang21        119.969600
dih21        179.782500
bnd22          1.078400
ang22        119.949300
dih22        179.800500
bnd23          1.007700
ang23        115.794300
dih23          4.358500
bnd24          1.078300
ang24        123.744000
dih24          0.095000
bnd25          1.088500
ang25        111.302900
dih25        117.185200
bnd26          1.088500
ang26        111.457600
dih26       -122.225000
bnd27          1.086700
ang27        109.579600
dih27         -2.466700
 constants:
dih7         -15.000000
dih8         105.000000
