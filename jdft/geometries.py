"""Geometry of the molecules."""

h2_geometry = """
H 0.0000 0.0000 0.0000;
H 0.0000 0.0000 0.7414;
"""

h2o_geometry = """
O 0.0000 0.0000 0.1173;
H 0.0000 0.7572 -0.4692;
H 0.0000 -0.7572 -0.4692;
"""

o2_geometry = """
O 0.0000 0.0000 0.0000;
O 0.0000 0.0000 1.2075;
"""

co2_geometry = """
C 0.0000 0.0000 0.0000
O 0.0000 0.0000 1.1621
O 0.0000 0.0000 -1.1621
"""

benzene_geometry = """
C 0.0000 1.3970 0.0000;
C 1.2098 0.6985 0.0000;
C 1.2098 -0.6985 0.0000;
C 0.0000 -1.3970 0.0000;
C -1.2098 -0.6985 0.0000;
C -1.2098 0.6985 0.0000;
H 0.0000 2.4810 0.0000;
H 2.1486 1.2405 0.0000;
H 2.1486 -1.2405 0.0000;
H 0.0000 -2.4810 0.0000;
H -2.1486 -1.2405 0.0000;
H -2.1486 1.2405 0.0000;
"""

n2_geometry = """
N 0.0000 0.0000 0.5488;
N 0.0000 0.0000 -0.5488;
"""

ch4_geometry = """
C 0.0000 0.0000 0.0000;
H 0.6276 0.6276 0.6276;
H 0.6276 -0.6276 -0.6276;
H -0.6276 0.6276 -0.6276;
H -0.6276 -0.6276 0.6276;
"""

hf_geometry = """
F 0.0000 0.0000 0.0000;
H 0.0000 0.0000 0.9168;
"""

ethonal_geometry = """
C 1.1879 -0.3829 0.0000;
C 0.0000 0.5526 0.0000;
O -1.1867 -0.2472 0.0000;
H -1.9237 0.3850 0.0000;
H 2.0985 0.2306 0.0000;
H 1.1184 -1.0093 0.8869;
H 1.1184 -1.0093 -0.8869;
H -0.0227 1.1812 0.8852;
H -0.0227 1.1812 -0.8852;
"""

c20_geometry = """
 C  1.56910 -0.65660 -0.93640;
 C  1.76690 0.64310 -0.47200;
 C  0.47050 -0.66520 -1.79270;
 C  0.01160 0.64780 -1.82550;
 C  0.79300 1.46730 -1.02840;
 C -0.48740 -1.48180 -1.21570;
 C -1.56350 -0.65720 -0.89520;
 C -1.26940 0.64900 -1.27670;
 C -0.00230 -1.96180 -0.00720;
 C -0.76980 -1.45320 1.03590;
 C -1.75760 -0.63800 0.47420;
 C  1.28780 -1.45030 0.16290;
 C  1.28960 -0.65950 1.30470;
 C  0.01150 -0.64600 1.85330;
 C  1.58300 0.64540 0.89840;
 C  0.48480 1.43830 1.19370;
 C -0.50320 0.64690 1.77530;
 C -1.60620 0.67150 0.92310;
 C -1.29590 1.48910 -0.16550;
 C -0.01020 1.97270 -0.00630;
 """

c36_geometry = """
 C     1.38000  -1.08960  -1.90050;
 C    -1.38280  -1.09260  -1.90290;
 C    -0.68460  -2.09080  -1.21930;
 C     0.70250  -2.10280  -1.20900;
 C     2.49770  -0.68080  -1.18920;
 C     2.50530   0.69670  -1.19290;
 C     2.48600   1.38910  -0.00630;
 C     2.48430  -1.38270  -0.00980;
 C     2.47240  -0.68540   1.18070;
 C     2.47510   0.69280   1.18450;
 C    -0.68990  -2.06790   1.20370;
 C     0.70280  -2.08960   1.21460;
 C     1.38890  -2.21450   0.00150;
 C    -1.38940  -2.16680  -0.01110;
 C     0.70940   2.09690   1.21910;
 C     1.38520   2.21100  -0.00690;
 C    -0.68010   2.08320   1.22070;
 C     1.38940   1.09930   1.92690;
 C     0.70560   2.09350  -1.22100;
 C     1.39350   1.09480  -1.91930;
 C    -0.68340   2.07520  -1.21360;
 C    -1.36740   2.16290   0.00260;
 C    -1.39730   1.09110  -1.90520;
 C    -0.69260   0.00560  -2.44080;
 C     0.69530   0.00550  -2.44600;
 C    -2.48590   1.35940  -0.01130;
 C    -2.51540   0.67740  -1.20860;
 C    -2.51620  -0.69790  -1.20030;
 C    -2.46990   0.68880   1.19780;
 C     0.69880   0.00210   2.46780;
 C    -0.69470  -0.00720   2.45580;
 C    -1.37180   1.09580   1.92640;
 C     1.38460  -1.09180   1.92070;
 C    -1.39770  -1.08670   1.90450;
 C    -2.51470  -0.69370   1.18910;
 C    -2.52570  -1.37800  -0.00150;
"""

c60_geometry = """
 C     2.16650   0.59060   2.58740;
 C     3.03780   0.17660   1.59180;
 C     1.27860  -0.30980   3.16790;
 C     3.01180  -1.14340   1.16540;
 C     3.10340  -1.43350  -0.19300;
 C     3.15030   1.21060   0.66820;
 C     3.24280   0.91490  -0.68590;
 C     3.21920  -0.40230  -1.12070;
 C    -0.43930   1.35270   3.12710;
 C     0.43630   2.26180   2.55420;
 C    -0.02960   0.06330   3.43790;
 C     1.74420   1.87900   2.28300;
 C     2.35190   2.26760   1.09900;
 C    -0.26330   3.02680   1.63260;
 C     0.33740   3.40540   0.43730;
 C     1.65160   3.02780   0.17070;
 C    -2.09030  -0.82250   2.59550;
 C    -2.51110   0.46640   2.28540;
 C    -0.84490  -1.02520   3.17380;
 C    -1.68740   1.55330   2.55120;
 C    -1.58430   2.58580   1.63190;
 C    -3.23140   0.40610   1.10070;
 C    -3.12270   1.44100   0.17460;
 C    -2.29470   2.52910   0.43990;
 C    -0.49080  -2.91330   1.73650;
 C    -1.74300  -2.71240   1.16370;
 C    -0.03930  -2.06840   2.74530;
 C    -2.54860  -1.66500   1.59420;
 C    -3.26020  -0.91410   0.67010;
 C    -1.65430  -3.00610  -0.18970;
 C    -2.35420  -2.24390  -1.11700;
 C    -3.16430  -1.19490  -0.68780;
 C     2.13640  -2.05530   1.73580;
 C     1.68950  -2.90090   0.72930;
 C     1.27850  -1.63660   2.74350;
 C     0.36780  -3.33270   0.73020;
 C    -0.34400  -3.39040  -0.45940;
 C     2.28890  -2.52500  -0.46400;
 C     1.57900  -2.57180  -1.65800;
 C     0.25600  -3.00540  -1.65310;
 C    -2.18280  -0.57830  -2.59790;
 C    -1.74800  -1.86940  -2.30830;
 C    -0.43850  -2.24690  -2.58450;
 C    -1.28150   0.31890  -3.16710;
 C    -2.15260   2.05450  -1.73780;
 C    -3.04850   1.15350  -1.18110;
 C    -3.06560  -0.16290  -1.61070;
 C    -1.26610   1.64070  -2.72710;
 C     0.50390   2.93610  -1.74180;
 C    -0.37880   3.35610  -0.75130;
 C    -1.69430   2.91860  -0.74910;
 C     0.05210   2.07300  -2.73550;
 C     2.09760   0.83400  -2.60510;
 C     2.55170   1.69230  -1.61070;
 C     1.75890   2.74520  -1.18240;
 C     0.84200   1.02060  -3.17860;
 C     0.44610  -1.34950  -3.16610;
 C     1.69830  -1.54850  -2.59080;
 C     2.51840  -0.46230  -2.31710;
 C     0.02180  -0.06450  -3.45850;
"""

c80_geometry = """
 C     2.27000   2.09900   2.48050;
 C     1.04360   2.26610   3.11140;
 C     0.52350   1.16990   3.78370;
 C     0.15120   3.10450   2.45790;
 C     0.45720   3.72900   1.25530;
 C     1.68080   3.55020   0.59820;
 C     2.58350   2.72300   1.27860;
 C    -1.22210   2.85270   2.47180;
 C    -1.73720   1.74810   3.13610;
 C    -0.84270   0.91510   3.79180;
 C    -2.83070   1.14750   2.52530;
 C    -0.72450   3.83480  -0.78430;
 C    -0.72570   3.87090   0.57000;
 C    -1.76490   3.32410   1.28230;
 C    -2.84400   2.71120   0.63880;
 C    -3.36820   1.61160   1.33020;
 C     2.57340   2.64730  -1.46460;
 C     1.67160   3.51140  -0.82490;
 C     0.44690   3.65570  -1.48770;
 C    -3.01360  -0.23750   2.56910;
 C    -2.10420  -1.06020   3.22190;
 C    -1.01650  -0.45900   3.83710;
 C    -1.88960  -2.29590   2.62460;
 C    -3.89010   0.48390  -0.67480;
 C    -3.88560   0.51730   0.67940;
 C    -3.66480  -0.62820   1.40470;
 C    -3.43550  -1.85560   0.77550;
 C    -2.53220  -2.68320   1.45440;
 C    -1.76360   3.23830  -1.46590;
 C    -2.84740   2.66280  -0.78400;
 C    -3.38060   1.53170  -1.41240;
 C    -0.62780  -2.89740   2.64070;
 C     0.45490  -2.27670   3.25170;
 C     0.23820  -1.04730   3.85620;
 C     1.67860  -2.46390   2.61900;
 C    -1.68260  -3.56450  -0.56290;
 C    -1.67210  -3.52540   0.79130;
 C    -0.49310  -3.65660   1.48390;
 C     0.72510  -3.82810   0.81980;
 C     1.81140  -3.21350   1.45570;
 C    -3.66220  -0.70110  -1.34140;
 C    -3.43220  -1.89870  -0.64730;
 C    -2.54120  -2.76670  -1.28730;
 C     2.64310  -1.45100   2.58880;
 C     2.40830  -0.21980   3.18840;
 C     1.19310  -0.04340   3.82790;
 C     2.94230   0.87380   2.51680;
 C     2.84880  -2.71180  -0.60270;
 C     2.85730  -2.66720   0.75100;
 C     3.36960  -1.57630   1.41010;
 C     3.89100  -0.48260   0.71160;
 C     3.66390   0.75180   1.33410;
 C    -0.50760  -3.72690  -1.26440;
 C     0.72320  -3.86390  -0.60350;
 C     1.80410  -3.29580  -1.28700;
 C     3.34530  -1.65510  -1.33570;
 C     3.88010  -0.51810  -0.71270;
 C     3.65310   0.67370  -1.41050;
 C     3.44840   1.86210  -0.74370;
 C     3.45440   1.89980   0.60760;
 C     2.61960  -1.60650  -2.52030;
 C     1.15970  -0.26320  -3.80050;
 C     2.39210  -0.39990  -3.17110;
 C     2.92780   0.74010  -2.59340;
 C    -0.64920  -3.05420  -2.47380;
 C     0.20480  -1.27410  -3.76980;
 C     0.43930  -2.47630  -3.11200;
 C     1.66890  -2.62310  -2.49540;
 C    -3.03420  -0.37900  -2.53950;
 C    -1.05060  -0.67640  -3.78280;
 C    -2.13540  -1.25630  -3.13470;
 C    -1.91110  -2.45850  -2.48630;
 C    -1.24080   2.71300  -2.64310;
 C    -0.87180   0.70260  -3.82620;
 C    -1.77530   1.57200  -3.22550;
 C    -2.86280   1.00180  -2.58810;
 C     2.26140   1.96190  -2.63370;
 C     0.49520   0.95820  -3.83780;
 C     1.02620   2.10110  -3.25040;
 C     0.13030   2.97330  -2.65650;
"""

c100_geometry = """
 C     4.99000   0.71220  -2.17150;
 C     4.20250  -0.20390  -2.81920;
 C     5.79810   0.36340  -1.10120;
 C     4.21630  -1.48540  -2.40760;
 C     3.01470  -2.15400  -2.41840;
 C    -0.60500  -0.33860  -3.44330;
 C    -0.60500  -1.72250  -2.99750;
 C     1.81860  -1.69670  -2.95760;
 C     1.81740  -0.34090  -3.39660;
 C     3.01100   0.34320  -3.22430;
 C     0.60650  -3.15370  -1.40850;
 C     0.60710  -2.29700  -2.58090;
 C    -3.01290  -3.16300  -0.69070;
 C    -1.81660  -3.10210  -1.39550;
 C    -1.81940  -2.26410  -2.54840;
 C    -3.01800  -1.61980  -2.81180;
 C     4.98000   2.28470   0.01340;
 C     4.20900   2.61670  -1.06350;
 C     5.79480   1.16370   0.02640;
 C     4.20470   1.83480  -2.15520;
 C     3.01230   1.64790  -2.80450;
 C    -0.60670   3.16180  -1.38170;
 C    -0.60430   2.31680  -2.56550;
 C     1.81820   2.29570  -2.52830;
 C     1.81570   3.12140  -1.36710;
 C     3.01170   3.16610  -0.66150;
 C     0.60810   0.36880  -3.44130;
 C     0.60880   1.74800  -2.98630;
 C    -3.01790  -0.31300  -3.23200;
 C    -1.81780   0.36870  -3.39200;
 C    -1.81890   1.72240  -2.94410;
 C    -3.01260   2.17560  -2.40370;
 C     4.99810   0.69100   2.20280;
 C     4.21260   1.81390   2.17640;
 C     5.80490   0.34140   1.13480;
 C     4.20150   2.60720   1.09210;
 C     3.00980   3.16570   0.71060;
 C    -0.60800   2.28880   2.59470;
 C    -0.60510   3.14860   1.42180;
 C     1.81630   3.10520   1.41200;
 C     1.81560   2.26090   2.55980;
 C     3.01370   1.60560   2.82300;
 C     0.60610   3.37850  -0.70390;
 C     0.60700   3.37130   0.74900;
 C    -3.01600   2.97280  -1.28740;
 C    -1.81670   3.33300  -0.68730;
 C    -1.81870   3.32380   0.73860;
 C    -3.01230   2.95000   1.33850;
 C     4.98360  -1.86000   1.35020;
 C     4.20330  -1.50260   2.41310;
 C     5.80710  -0.96460   0.68530;
 C     4.20710  -0.22690   2.84140;
 C     3.01320   0.30440   3.25380;
 C    -0.60710  -1.75660   2.99440;
 C    -0.60790  -0.37700   3.45390;
 C     1.81570  -0.37640   3.41440;
 C     1.81500  -1.72870   2.96060;
 C     3.00950  -2.18020   2.41500;
 C     0.60470   1.71270   3.01190;
 C     0.60490   0.33120   3.45920;
 C    -3.01900   2.14350   2.44870;
 C    -1.81860   1.68450   2.97630;
 C    -1.82220   0.32600   3.41120;
 C    -3.02230  -0.35030   3.24110;
 C     5.00820  -1.86700  -1.36150;
 C     4.21900  -2.75720  -0.68060;
 C     5.81340  -0.96470  -0.69810;
 C     4.20410  -2.75560   0.66480;
 C     3.00950  -2.98370   1.30280;
 C    -0.60700  -3.37230  -0.73300;
 C    -0.60700  -3.38540   0.72030;
 C     1.81450  -3.34660   0.70020;
 C     1.81940  -3.33830  -0.72430;
 C     3.01780  -2.97280  -1.31890;
 C     0.60590  -2.32690   2.57530;
 C     0.60600  -3.17250   1.39430;
 C    -3.01870  -1.65160   2.80190;
 C    -1.81870  -2.29670   2.53280;
 C    -1.82170  -3.12800   1.37580;
 C    -3.02000  -3.17720   0.68000;
 C    -4.99260  -0.72440   2.18890;
 C    -4.21450  -1.85200   2.15470;
 C    -4.22280  -2.63690   1.06150;
 C    -5.79260  -0.37960   1.11070;
 C    -4.97750   1.84280   1.37470;
 C    -4.21690   1.47200   2.44400;
 C    -4.21950   0.19510   2.84670;
 C    -5.77810   0.94260   0.68970;
 C    -4.97930   1.86080  -1.34630;
 C    -4.21200   2.75560  -0.64930;
 C    -4.20900   2.74460   0.69090;
 C    -5.78610   0.94390  -0.69140;
 C    -4.99790  -0.70290  -2.19610;
 C    -4.21670   0.22660  -2.83450;
 C    -4.21050   1.49940  -2.41340;
 C    -5.80510  -0.37120  -1.11870;
 C    -5.00870  -2.31690  -0.01080;
 C    -4.21620  -2.62180  -1.08820;
 C    -4.21470  -1.82790  -2.17170;
 C    -5.81450  -1.19590  -0.00730;
"""

c180_geometry = """
 C     5.94060   0.59650  -0.41930;
 C     5.93570  -0.73530  -0.16200;
 C     5.66630  -1.38200  -1.32280;
 C     5.50750  -0.45190  -2.29460;
 C     5.67810   0.77110  -1.73860;
 C     2.83010   5.12760   1.23450;
 C     3.83470   4.56120   0.52130;
 C     3.57290   4.73550  -0.79620;
 C     2.39920   5.40580  -0.89720;
 C     1.94210   5.65200   0.35430;
 C     3.37000  -0.17130   4.94290;
 C     4.38070   0.08380   4.07510;
 C     4.38280   1.41520   3.81810;
 C     3.37330   1.97720   4.52830;
 C     2.75020   0.99860   5.22730;
 C     2.37120  -5.36250   1.18010;
 C     3.54640  -4.70310   1.03000;
 C     3.81640  -4.05840   2.19080;
 C     2.80660  -4.31830   3.05820;
 C     1.91470  -5.12670   2.43360;
 C     1.21740  -3.27530  -4.86340;
 C     2.49140  -3.19300  -4.40510;
 C     2.65450  -4.12300  -3.43440;
 C     1.47620  -4.77650  -3.28620;
 C     0.59020  -4.25790  -4.16960;
 C     1.49520   3.20730  -4.82930;
 C     2.66790   2.53390  -4.72220;
 C     2.50350   1.31170  -5.28280;
 C     1.22700   1.23160  -5.73180;
 C     0.60550   2.40280  -5.45710;
 C    -5.93570  -0.59440   0.42270;
 C    -5.67490  -0.76940   1.74010;
 C    -5.50630   0.45710   2.29310;
 C    -5.66940   1.38700   1.32150;
 C    -5.93590   0.73720   0.16170;
 C    -1.94110  -5.64480  -0.34860;
 C    -2.40080  -5.40690   0.90320;
 C    -3.57220  -4.73310   0.79580;
 C    -3.83120  -4.55720  -0.52420;
 C    -2.82610  -5.12390  -1.23370;
 C    -0.60830  -2.40370   5.45100;
 C    -1.22900  -1.22950   5.72670;
 C    -2.50360  -1.31650   5.27460;
 C    -2.66790  -2.54160   4.72150;
 C    -1.49840  -3.21580   4.83180;
 C    -0.59120   4.25700   4.16640;
 C    -1.47730   4.78080   3.28550;
 C    -2.65030   4.11680   3.43640;
 C    -2.48640   3.19020   4.41060;
 C    -1.21440   3.27630   4.86590;
 C    -1.91330   5.12020  -2.42570;
 C    -2.80240   4.31240  -3.05500;
 C    -3.81120   4.05680  -2.18500;
 C    -3.54830   4.70870  -1.02640;
 C    -2.37370   5.36730  -1.17590;
 C    -2.75150  -1.00080  -5.22350;
 C    -3.37950  -1.97970  -4.52910;
 C    -4.38810  -1.41580  -3.82380;
 C    -4.38430  -0.08480  -4.08200;
 C    -3.37290   0.17320  -4.94870;
 C     5.57710   1.08350   1.83600;
 C     4.43850   3.23390   2.34680;
 C     5.28420   2.79100   0.04890;
 C     4.93750   1.97320   2.71950;
 C     5.74720   1.54940   0.51980;
 C     4.65680   3.60500   1.00830;
 C     5.57430  -0.34890   2.11230;
 C     4.42640  -2.15060   3.39110;
 C     5.26880  -2.59390   1.09270;
 C     4.93250  -0.84460   3.26330;
 C     5.74000  -1.27030   1.06310;
 C     4.64190  -2.99270   2.28460;
 C     4.98110  -3.29150  -0.15470;
 C     3.34730  -4.77230  -1.30340;
 C     4.49930  -2.97310  -2.57300;
 C     4.07280  -4.36590  -0.16980;
 C     5.17630  -2.64090  -1.38770;
 C     3.61360  -4.06120  -2.48620;
 C     4.32560  -1.97330  -3.62020;
 C     2.69660  -1.01610  -5.23840;
 C     4.33340   0.46670  -4.09490;
 C     3.26990  -2.09560  -4.54410;
 C     4.83730  -0.67480  -3.44870;
 C     3.27860   0.24260  -4.99800;
 C     5.00110   2.97520  -1.36880;
 C     3.37090   3.93170  -2.98480;
 C     4.51340   1.78380  -3.49510;
 C     4.09730   3.97270  -1.78120;
 C     5.19190   1.91430  -2.27160;
 C     3.63030   2.82940  -3.81860;
 C     3.35530   3.83830   3.11290;
 C     1.50250   3.36900   4.70820;
 C     1.20710   5.07450   2.91980;
 C     2.80700   3.16710   4.22090;
 C     2.52450   4.80090   2.51020;
 C     0.75140   4.35030   4.03430;
 C     0.26130   5.69300  -1.79430;
 C    -0.22800   4.50100  -3.92490;
 C     2.11060   4.65470  -3.09810;
 C    -0.61880   5.24320  -2.79750;
 C     1.62070   5.39090  -2.00420;
 C     1.15220   4.25230  -4.04230;
 C     0.25520   5.63800   1.96900;
 C    -2.08430   5.48330   1.14000;
 C    -0.23060   5.95560  -0.44780;
 C    -1.12660   5.45600   2.16810;
 C     0.64870   5.90900   0.64650;
 C    -1.59020   5.76060  -0.14830;
 C     3.34110  -2.42890   4.32250;
 C     1.18430  -3.63500   4.60310;
 C     1.49470  -1.38840   5.62330;
 C     2.50300  -3.54070   4.12210;
 C     2.79910  -1.38800   5.09960;
 C     0.73410  -2.54270   5.36710;
 C     0.82690  -0.12770   5.92340;
 C    -1.32370   1.10600   5.72560;
 C     0.83250   2.31330   5.45590;
 C    -0.57910  -0.06470   5.95260;
 C     1.48580   1.09320   5.69450;
 C    -0.57150   2.27530   5.50530;
 C     2.08400  -5.47980  -1.13790;
 C    -0.25530  -5.63370  -1.96610;
 C     0.23070  -5.95420   0.45400;
 C     1.12480  -5.45150  -2.16760;
 C     1.58950  -5.75730   0.14900;
 C    -0.64750  -5.90410  -0.64310;
 C    -0.26090  -5.69470   1.80130;
 C    -2.11070  -4.65830   3.10220;
 C     0.22940  -4.50830   3.93000;
 C    -1.62110  -5.39420   2.00760;
 C     0.62070  -5.25020   2.80300;
 C    -1.14910  -4.25880   4.04770;
 C     1.32480  -1.10600  -5.72510;
 C    -0.82670   0.12640  -5.92430;
 C    -0.83270  -2.31670  -5.45360;
 C     0.57880   0.06580  -5.95410;
 C     0.57250  -2.27350  -5.50360;
 C    -1.48540  -1.09440  -5.69230;
 C    -1.20610  -5.07110  -2.91790;
 C    -3.35630  -3.83580  -3.11560;
 C    -1.50400  -3.37190  -4.70700;
 C    -2.52500  -4.79620  -2.51020;
 C    -0.75120  -4.34940  -4.03600;
 C    -2.80860  -3.16690  -4.22480;
 C    -1.18410   3.63010  -4.59950;
 C    -3.34400   2.42360  -4.32360;
 C    -1.49450   1.38780  -5.62530;
 C    -2.50400   3.53340  -4.11940;
 C    -0.73450   2.54190  -5.36800;
 C    -2.79830   1.38680  -5.10360;
 C    -5.28210  -2.78640  -0.04890;
 C    -4.43990  -3.22970  -2.34990;
 C    -5.58160  -1.08230  -1.83860;
 C    -4.65460  -3.60040  -1.01090;
 C    -5.74710  -1.54440  -0.52180;
 C    -4.94350  -1.96950  -2.72300;
 C    -4.99750  -2.97540   1.36780;
 C    -3.37150  -3.93370   2.98690;
 C    -4.51240  -1.78380   3.49440;
 C    -4.09640  -3.97220   1.78180;
 C    -5.18950  -1.91320   2.27080;
 C    -3.63200  -2.83060   3.81940;
 C    -4.33190  -0.46660   4.09240;
 C    -2.69540   1.01560   5.23780;
 C    -4.32420   1.97410   3.62040;
 C    -3.27600  -0.24160   4.99570;
 C    -4.83560   0.67700   3.44730;
 C    -3.27010   2.09710   4.54330;
 C    -4.49720   2.97340   2.57230;
 C    -3.34830   4.77430   1.30130;
 C    -4.98450   3.29150   0.15260;
 C    -3.60960   4.06190   2.48420;
 C    -5.17700   2.64390   1.38570;
 C    -4.07750   4.36590   0.16870;
 C    -5.26910   2.59410  -1.09490;
 C    -4.42920   2.14530  -3.39250;
 C    -5.57870   0.34880  -2.11590;
 C    -4.63860   2.99100  -2.28840;
 C    -5.74200   1.26840  -1.06660;
 C    -4.93810   0.84050  -3.26730;
"""
