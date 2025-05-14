echo Rungauss
%mem=14000MB
%nprocshared=  16
%nosave
%chk=roy_struc_1.chk
#MP2/6-311G(d,p) Opt=(Z-matrix,MaxCycle=500,MaxStep=20,VeryTight) int=ultrafine SCF=QC
#Pop=hlygat nosymm

roy struc    1

0 1
S
C    1    bnd2
C    2    bnd3    1    ang3
C    3    bnd4    2    ang4    1    dih4
C    4    bnd5    3    ang5    2    dih5
N    2    bnd6    3    ang6    4    dih6
C    6    bnd7    2    ang7    3    dih7
C    7    bnd8    6    ang8    2    dih8
C    8    bnd9    7    ang9    6    dih9
C    9    bnd10    8    ang10    7    dih10
C    10    bnd11    9    ang11    8    dih11
C    11    bnd12    10    ang12    9    dih12
N    8    bnd13    7    ang13    6    dih13
O    13    bnd14    8    ang14    7    dih14
C    5    bnd15    4    ang15    3    dih15
C    3    bnd16    2    ang16    6    dih16
N    16    bnd17    2    ang17    3    dih17
O    13    bnd18    8    ang18    7    dih18
H    9    bnd19    8    ang19    7    dih19
H    10    bnd20    9    ang20    8    dih20
H    11    bnd21    10    ang21    9    dih21
H    12    bnd22    11    ang22    10    dih22
H    6    bnd23    7    ang23    8    dih23
H    4    bnd24    5    ang24    15    dih24
H    15    bnd25    5    ang25    4    dih25
H    15    bnd26    5    ang26    4    dih26
H    15    bnd27    5    ang27    4    dih27
Variables:
bnd2     1.739251
bnd3     1.395208
ang3     110.214483
bnd4     1.430972
ang4     112.455572
dih4     -3.596750
bnd5     1.374291
ang5     113.584055
dih5     1.493897
bnd6     1.380065
ang6     131.155617
dih6     -174.291772
bnd7     1.384074
ang7     128.149356
dih7     30.000000
bnd8     1.422477
ang8     121.751045
bnd9     1.399689
dih8     180.000000
ang9     121.730905
dih9     -177.175882
bnd10     1.389475
ang10     120.305889
dih10     5.836940
bnd11     1.398407
ang11     118.976315
dih11     -1.260221
bnd12     1.388813
ang12     120.742602
dih12     -0.146007
bnd13     1.468022
ang13     122.002399
dih13     9.288278
bnd14     1.228876
ang14     118.443591
dih14     167.112212
bnd15     1.498870
ang15     127.934854
dih15     -176.596632
bnd16     1.421603
ang16     123.956975
dih16     16.827958
bnd17     1.180399
ang17     152.786658
dih17     -179.786646
bnd18     1.242328
ang18     118.562799
dih18     -14.006365
bnd19     1.084346
ang19     118.093249
dih19     -176.270244
bnd20     1.083782
ang20     119.952986
dih20     179.409734
bnd21     1.088203
ang21     120.454115
dih21     177.804163
bnd22     1.081267
ang22     119.278347
dih22     176.293971
bnd23     1.017281
ang23     113.426516
dih23     -2.935056
bnd24     1.083864
ang24     123.285138
dih24     1.205453
bnd25     1.093860
ang25     111.322256
dih25     115.934568
bnd26     1.092750
ang26     111.706336
dih26     -123.077921
bnd27     1.094709
ang27     109.067784
dih27     -3.466637
