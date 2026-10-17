echo Rungauss
%chk=uniform_lam_00001.chk
%mem=14000MB
%nprocshared=8
# HF/3-21G Freq NoSymm

HF Cartesian Hessian at fixed geometry
 
 0 1
C1
C1   1 bnd2  
C1   2 bnd3     1 ang3  
C1   3 bnd4     2 ang4     1 dih4  
C1   4 bnd5     3 ang5     2 dih5  
N1   1 bnd6     2 ang6     3 dih6  
C1   5 bnd7     6 ang7     1 dih7  
C1   7 bnd8     5 ang8     6 dih8  
C1   8 bnd9     7 ang9     5 dih9  
N1   9 bnd10    8 ang10    7 dih10 
N1  10 bnd11    9 ang11    8 dih11 
C1  11 bnd12   10 ang12    9 dih12 
C1   9 bnd13    8 ang13   10 dih13 
C1  13 bnd14    9 ang14   10 dih14 
C1  14 bnd15   13 ang15    9 dih15 
C1  12 bnd16   11 ang16   10 dih16 
C1  15 bnd17   14 ang17   13 dih17 
S1  17 bnd18   15 ang18   14 dih18 
C1  18 bnd19   17 ang19   15 dih19 
C1  19 bnd20   18 ang20   17 dih20 
C1  20 bnd21   19 ang21   18 dih21 
C1  21 bnd22   20 ang22   19 dih22 
C1  22 bnd23   21 ang23   20 dih23 
C1  19 bnd24   18 ang24   20 dih24 
C1  24 bnd25   19 ang25   20 dih25 
N1  25 bnd26   24 ang26   23 dih26 
O1  25 bnd27   24 ang27   26 dih27 
C1  26 bnd28   25 ang28   24 dih28 
H1  28 bnd29   26 ang29   25 dih29 
H1  28 bnd30   26 ang30   29 dih30 
H1  28 bnd31   26 ang31   29 dih31 
H2  26 bnd32   25 ang32   28 dih32 
H1  23 bnd33   24 ang33   19 dih33 
H1  22 bnd34   23 ang34   24 dih34 
H1  21 bnd35   22 ang35   23 dih35 
H1  20 bnd36   21 ang36   22 dih36 
H1  15 bnd37   14 ang37   13 dih37 
H1  14 bnd38   13 ang38   12 dih38 
H1  16 bnd39   17 ang39   15 dih39 
H2  11 bnd40   10 ang40    9 dih40 
H1   8 bnd41    7 ang41    5 dih41 
H1   7 bnd42    5 ang42    8 dih42 
H1   4 bnd43    3 ang43    2 dih43 
H1   3 bnd44    2 ang44    1 dih44 
H1   2 bnd45    1 ang45    6 dih45 
H1   1 bnd46    6 ang46    5 dih46 
 variables:
bnd2           1.388600
bnd3           1.384100
ang3         118.028600
bnd4           1.383400
ang4         118.700700
dih4          -0.005300
bnd5           1.394300
ang5         119.295200
dih5          -0.005100
bnd6           1.325700
ang6         123.864200
dih6          -0.000100
bnd7           1.470700
ang7         118.390900
dih7         179.943800
bnd8           1.332300
ang8         123.547300
bnd9           1.454600
ang9         126.024300
dih9        -179.993400
bnd10          1.318600
ang10        118.284200
bnd11          1.345300
ang11        107.110200
dih11        179.741000
bnd12          1.363200
ang12        112.292200
dih12          0.087100
bnd13          1.442500
ang13        131.619200
dih13        179.697500
bnd14          1.402400
ang14        136.977200
dih14        178.815300
bnd15          1.374400
ang15        119.133200
dih15       -178.094100
bnd16          1.396100
ang16        131.033100
dih16        179.992700
bnd17          1.409600
ang17        121.104200
dih17         -1.293700
bnd18          1.775900
ang18        118.154400
dih18       -179.280000
bnd19          1.776700
ang19        100.603400
bnd20          1.389700
ang20        117.703800
bnd21          1.387400
ang21        120.702000
dih21       -179.056500
bnd22          1.384600
ang22        119.864400
dih22         -1.588900
bnd23          1.387700
ang23        119.644200
dih23          0.918400
bnd24          1.397800
ang24        122.761700
dih24       -179.397800
bnd25          1.508000
ang25        122.035600
dih25       -175.795900
bnd26          1.358000
ang26        114.051800
bnd27          1.214500
ang27        122.183200
dih27       -177.455500
bnd28          1.448800
ang28        122.258400
bnd29          1.084800
ang29        108.086500
dih29         21.622500
bnd30          1.089500
ang30        111.299900
dih30       -119.422300
bnd31          1.087500
ang31        110.066400
dih31        119.258400
bnd32          1.003000
ang32        117.151300
dih32       -163.710000
bnd33          1.081100
ang33        118.908900
dih33        179.791600
bnd34          1.080000
ang34        119.955300
dih34       -178.619100
bnd35          1.080400
ang35        120.346000
dih35        179.920900
bnd36          1.080400
ang36        120.369000
dih36        177.716900
bnd37          1.078800
ang37        120.466500
dih37        176.434900
bnd38          1.078100
ang38        121.496700
dih38        178.743100
bnd39          1.079900
ang39        121.015700
dih39       -179.423300
bnd40          1.001300
ang40        119.105500
dih40        178.207300
bnd41          1.081500
ang41        118.656100
dih41         -0.217500
bnd42          1.080200
ang42        115.415200
dih42       -179.621000
bnd43          1.080300
ang43        121.000500
dih43       -179.968300
bnd44          1.080600
ang44        120.813700
dih44       -179.997200
bnd45          1.079400
ang45        120.360500
dih45        179.997900
bnd46          1.082800
ang46        116.003200
dih46       -179.992900
 constants:
dih8           0.000000
dih10          0.000000
dih19       -130.000000
dih20       -135.000000
dih26       -120.000000
dih28        180.000000
 
