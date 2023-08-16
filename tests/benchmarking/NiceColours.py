"""
Python library for CVD-friendly ("colour blind") pallette
- by Tim Gould

Colours from https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/

It is not a proper library...

Use:

NiceColour("Beige")

or

NiceColour(5)

If you increment the pallette by integers it should be easy to
distinguish for 95% of people. Alternately, if you run
BestColour or NiceColour with Best=True you get a pallette of
ten that can be seen by 99% of people.
 
NiceColour(5, Best=True)
BestColour(4)

Optionally (but less recommended) is to run as

NiceColour("Beige", Shade=0.5)

which will make the colour darker. There is also 

NiceCMap()

which is just to my (TG) preference and only loosely based on
colour theory.

UPDATE 16TH OCT:
* Nicecolours now provides an interface to colorcet (if available)
  through NiceCmap: call NiceCMap(ID) when ID in
  ["fire", "rainbow", "coolwarm", "bmy","gwv",]
"""


# For making text clearer
import matplotlib.patheffects as peff
def AddBorder(txt, w=3, fg='w'):
    txt.set_path_effects([peff.withStroke(linewidth=w, foreground=fg)])


###############################################################
#
# This list is from an older version of
# https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
#
###############################################################

NiceColourTableData="""Red	#e6194b	(230, 25, 75)	(0, 100, 66, 0)
Green	#3cb44b	(60, 180, 75)	(75, 0, 100, 0)
Yellow	#ffe119	(255, 225, 25)	(0, 25, 95, 0)
Blue	#0082c8	(0, 130, 200)	(100, 35, 0, 0)
Orange	#f58231	(245, 130, 48)	(0, 60, 92, 0)
Purple	#911eb4	(145, 30, 180)	(35, 70, 0, 0)
Cyan	#46f0f0	(70, 240, 240)	(70, 0, 0, 0)
Magenta	#f032e6	(240, 50, 230)	(0, 100, 0, 0)
Lime	#d2f53c	(210, 245, 60)	(35, 0, 100, 0)
Pink	#fabebe	(250, 190, 190)	(0, 30, 15, 0)
Teal	#008080	(0, 128, 128)	(100, 0, 0, 50)
Lavender	#e6beff	(230, 190, 255)	(10, 25, 0, 0)
Brown	#aa6e28	(170, 110, 40)	(0, 35, 75, 33)
Beige	#fffac8	(255, 250, 200)	(5, 10, 30, 0)
Maroon	#800000	(128, 0, 0)	(0, 100, 100, 50)
Mint	#aaffc3	(170, 255, 195)	(33, 0, 23, 0)
Olive	#808000	(128, 128, 0)	(0, 0, 100, 50)
Coral	#ffd8b1	(255, 215, 180)	(0, 15, 30, 0)
Navy	#000080	(0, 0, 128)	(100, 100, 0, 50)
Grey	#808080	(128, 128, 128)	(0, 0, 0, 50)
White	#FFFFFF	(255, 255, 255)	(0, 0, 0, 0)
Black	#000000	(0, 0, 0)	(0, 0, 0, 100)"""

NiceColourNumber=22
NiceColourTable={}
NiceColourHex={}
NiceColourID=[]
BestColourID=("Maroon", "Lavender", "Navy", "Orange", "Blue",
              "Yellow", "Pink", "Black", "Grey", "White")
DarkColourID=(
    "Green", "Blue", "Orange", "Purple",
    "Magenta", "Teal", "Lavender", "Brown",
    "Beige", "Maroon", "Mint", "Olive",
    "Navy", "Grey", "White", "Black",)

def SetupNiceColours():
    for L in NiceColourTableData.split("\n"):
        D=L.split("\t")
        ID=D[0]
        TT=(D[2][1:-1]).split(",")

        NiceColourID.append(ID)
        NiceColourHex[ID]=D[1][1:]
        NiceColourTable[ID]=(int(TT[0])/255.,int(TT[1])/255.,int(TT[2])/255.)

def SetupNiceGreys():
    for L in NiceColourTableData.split("\n"):
        D=L.split("\t")
        ID=D[0]
        TT=(D[2][1:-1]).split(",")

        NiceColourID.append(ID)
        grey_lin=(0.2126*float(TT[0]) + 0.7152*float(TT[1])
                  + 0.0722*float(TT[2]))/255.
        if grey_lin <=0.0031308:
            grey = 12.92*grey_lin
        else:
            grey = 1.055*grey_lin**(1./2.4) - 0.055
            
        NiceColourTable[ID]=(grey,grey,grey)
    
        
def AllNiceColours():
    # Setup the first time called
    if not("Red" in NiceColourTable):
        SetupNiceColours()

    return NiceColourID

def NiceShade(rgb, S):
    return (rgb[0]*S, rgb[1]*S, rgb[2]*S)

def NiceColour(k=0, ID=None, Shade=1.0, Best=False):
    # Setup the first time called
    if not("Red" in NiceColourTable):
        SetupNiceColours()

    if ID is None:
        if isinstance(k,str):
            return NiceShade(NiceColourTable[k], Shade)
        else:
            if not(Best):
                return NiceShade(NiceColourTable[
                    NiceColourID[k%NiceColourNumber]]
                                 ,Shade)
            else:
                return NiceShade(NiceColourTable[BestColourID[k%10]],Shade)
    else:
        return NiceShade(NiceColourTable[ID], Shade)

def BestColour(k=0, ID=None, Shade=1.0):
    return NiceColour(k, ID, Shade, Best=True)


def DarkColour(k=0):
    return NiceColour(DarkColourID[k%12])

    

def MixColours(C1, C2):
    return ((C1[0]+C2[0])/2., (C1[1]+C2[1])/2., (C1[2]+C2[2])/2.)

from matplotlib.colors import LinearSegmentedColormap
try:
    import colorcet as cc
except:
    cc = None
    
def NiceCMap(ID="inferno"):
    if not(cc is None):
        # ["fire", "rainbow", "coolwarm", "bmy","gwv",]
        if ID.lower()=="fire": return cc.cm.fire
        elif ID.lower()=="rainbow": return cc.cm.rainbow
        elif ID.lower()=="coolwarm": return cc.cm.coolwarm
        elif ID.lower()=="bmy": return cc.cm.bmy
        elif ID.lower()=="gwv": return cc.cm.gwv
        
    if ID=="inferno":
        return LinearSegmentedColormap.from_list( \
            "inferno", [NiceColour(ID="Navy"),
                        #(0.92,0.66,0.33),
                        NiceColour(ID="Orange"),
                        NiceColour(ID="Beige")], N=100)
    elif ID=="gn_inferno":
        return LinearSegmentedColormap.from_list( \
            "inferno", [NiceColour(ID="Green"),
                        NiceColour(ID="Teal"),
                        NiceColour(ID="Navy"),
                        #(0.92,0.66,0.33),
                        NiceColour(ID="Orange"),
                        NiceColour(ID="Beige")], N=100)
    elif ID=="NaOr":
        return LinearSegmentedColormap.from_list\
    ("NaOr", [
  ( 0.00000,  0.00000,  0.50196),  ( 0.06381,  0.06381,  0.53374),  ( 0.10376,  0.10376,  0.55364),
  ( 0.13684,  0.13684,  0.57011),  ( 0.16693,  0.16693,  0.58510),  ( 0.19513,  0.19513,  0.59914),
  ( 0.22203,  0.22203,  0.61254),  ( 0.24802,  0.24802,  0.62548),  ( 0.27333,  0.27333,  0.63809),
  ( 0.29812,  0.29812,  0.65044),  ( 0.32251,  0.32251,  0.66258),  ( 0.34657,  0.34657,  0.67457),
  ( 0.37036,  0.37036,  0.68641),  ( 0.39393,  0.39393,  0.69815),  ( 0.41732,  0.41732,  0.70980),
  ( 0.44054,  0.44054,  0.72137),  ( 0.46363,  0.46363,  0.73287),  ( 0.48660,  0.48660,  0.74431),
  ( 0.50947,  0.50947,  0.75570),  ( 0.53224,  0.53224,  0.76704),  ( 0.55494,  0.55494,  0.77834),
  ( 0.57757,  0.57757,  0.78961),  ( 0.60013,  0.60013,  0.80085),  ( 0.62263,  0.62263,  0.81205),
  ( 0.64507,  0.64507,  0.82323),  ( 0.66748,  0.66748,  0.83439),  ( 0.68984,  0.68984,  0.84553),
  ( 0.71216,  0.71216,  0.85664),  ( 0.73445,  0.73445,  0.86774),  ( 0.75671,  0.75671,  0.87883),
  ( 0.77893,  0.77893,  0.88990),  ( 0.80113,  0.80113,  0.90096),  ( 0.82331,  0.82331,  0.91200),
  ( 0.84546,  0.84546,  0.92303),  ( 0.86759,  0.86759,  0.93405),  ( 0.88970,  0.88970,  0.94507),
  ( 0.91179,  0.91179,  0.95607),  ( 0.93387,  0.93387,  0.96706),  ( 0.95592,  0.95592,  0.97805),
  ( 0.97797,  0.97797,  0.98903),  ( 1.00000,  1.00000,  1.00000),  ( 0.99914,  0.98928,  0.98225),
  ( 0.99828,  0.97852,  0.96443),  ( 0.99742,  0.96772,  0.94654),  ( 0.99655,  0.95688,  0.92859),
  ( 0.99568,  0.94599,  0.91057),  ( 0.99481,  0.93507,  0.89247),  ( 0.99393,  0.92409,  0.87430),
  ( 0.99305,  0.91307,  0.85604),  ( 0.99216,  0.90199,  0.83770),  ( 0.99127,  0.89086,  0.81927),
  ( 0.99037,  0.87968,  0.80074),  ( 0.98947,  0.86843,  0.78212),  ( 0.98857,  0.85712,  0.76340),
  ( 0.98766,  0.84576,  0.74457),  ( 0.98675,  0.83432,  0.72564),  ( 0.98583,  0.82282,  0.70659),
  ( 0.98490,  0.81124,  0.68741),  ( 0.98397,  0.79959,  0.66811),  ( 0.98303,  0.78785,  0.64868),
  ( 0.98208,  0.77603,  0.62911),  ( 0.98113,  0.76412,  0.60939),  ( 0.98017,  0.75212,  0.58951),
  ( 0.97920,  0.74001,  0.56946),  ( 0.97822,  0.72780,  0.54924),  ( 0.97724,  0.71548,  0.52883),
  ( 0.97624,  0.70303,  0.50822),  ( 0.97524,  0.69046,  0.48740),  ( 0.97422,  0.67775,  0.46635),
  ( 0.97319,  0.66490,  0.44507),  ( 0.97215,  0.65189,  0.42352),  ( 0.97110,  0.63871,  0.40170),
  ( 0.97003,  0.62535,  0.37958),  ( 0.96894,  0.61180,  0.35714),  ( 0.96784,  0.59804,  0.33436),
  ( 0.96672,  0.58406,  0.31121),  ( 0.96559,  0.56982,  0.28761),  ( 0.96442,  0.55529,  0.26355),
  ( 0.96324,  0.54046,  0.23900),  ( 0.96202,  0.52531,  0.21391),  ( 0.96078,  0.50980,  0.18824),
], N=81)     
    elif ID=="cividis":
        # from https://github.com/pnnl/cmaputil
        return LinearSegmentedColormap.from_list( \
            "cividis", [
                ( 0.0000, 0.1262, 0.3015 ), # 0
                ( 0.0000, 0.1350, 0.3205 ), # 1
                ( 0.0000, 0.1437, 0.3400 ), # 2
                ( 0.0000, 0.1519, 0.3606 ), # 3
                ( 0.0000, 0.1601, 0.3817 ), # 4
                ( 0.0000, 0.1685, 0.4031 ), # 5
                ( 0.0000, 0.1773, 0.4241 ), # 6
                ( 0.0000, 0.1834, 0.4363 ), # 7
                ( 0.0000, 0.1901, 0.4365 ), # 8
                ( 0.0000, 0.1987, 0.4349 ), # 9
                ( 0.0000, 0.2073, 0.4329 ), # 10
                ( 0.0416, 0.2158, 0.4308 ), # 11
                ( 0.0827, 0.2244, 0.4287 ), # 12
                ( 0.1120, 0.2329, 0.4268 ), # 13
                ( 0.1359, 0.2414, 0.4251 ), # 14
                ( 0.1566, 0.2498, 0.4236 ), # 15
                ( 0.1752, 0.2583, 0.4224 ), # 16
                ( 0.1923, 0.2667, 0.4214 ), # 17
                ( 0.2082, 0.2751, 0.4207 ), # 18
                ( 0.2232, 0.2836, 0.4203 ), # 19
                ( 0.2375, 0.2920, 0.4200 ), # 20
                ( 0.2511, 0.3004, 0.4201 ), # 21
                ( 0.2643, 0.3088, 0.4203 ), # 22
                ( 0.2770, 0.3172, 0.4208 ), # 23
                ( 0.2894, 0.3256, 0.4215 ), # 24
                ( 0.3014, 0.3340, 0.4224 ), # 25
                ( 0.3132, 0.3424, 0.4236 ), # 26
                ( 0.3247, 0.3509, 0.4249 ), # 27
                ( 0.3361, 0.3593, 0.4264 ), # 28
                ( 0.3472, 0.3678, 0.4282 ), # 29
                ( 0.3582, 0.3763, 0.4302 ), # 30
                ( 0.3691, 0.3848, 0.4322 ), # 31
                ( 0.3798, 0.3933, 0.4346 ), # 32
                ( 0.3905, 0.4018, 0.4372 ), # 33
                ( 0.4010, 0.4104, 0.4400 ), # 34
                ( 0.4114, 0.4189, 0.4430 ), # 35
                ( 0.4218, 0.4275, 0.4462 ), # 36
                ( 0.4320, 0.4362, 0.4496 ), # 37
                ( 0.4422, 0.4448, 0.4534 ), # 38
                ( 0.4523, 0.4535, 0.4575 ), # 39
                ( 0.4622, 0.4622, 0.4620 ), # 40
                ( 0.4722, 0.4709, 0.4665 ), # 41
                ( 0.4825, 0.4797, 0.4701 ), # 42
                ( 0.4934, 0.4886, 0.4719 ), # 43
                ( 0.5045, 0.4975, 0.4730 ), # 44
                ( 0.5158, 0.5065, 0.4736 ), # 45
                ( 0.5272, 0.5155, 0.4739 ), # 46
                ( 0.5387, 0.5246, 0.4739 ), # 47
                ( 0.5502, 0.5338, 0.4735 ), # 48
                ( 0.5618, 0.5430, 0.4729 ), # 49
                ( 0.5735, 0.5522, 0.4720 ), # 50
                ( 0.5852, 0.5615, 0.4709 ), # 51
                ( 0.5970, 0.5709, 0.4696 ), # 52
                ( 0.6089, 0.5803, 0.4680 ), # 53
                ( 0.6208, 0.5898, 0.4662 ), # 54
                ( 0.6328, 0.5993, 0.4641 ), # 55
                ( 0.6449, 0.6089, 0.4617 ), # 56
                ( 0.6570, 0.6185, 0.4591 ), # 57
                ( 0.6691, 0.6282, 0.4562 ), # 58
                ( 0.6813, 0.6380, 0.4532 ), # 59
                ( 0.6936, 0.6478, 0.4499 ), # 60
                ( 0.7060, 0.6577, 0.4463 ), # 61
                ( 0.7184, 0.6676, 0.4424 ), # 62
                ( 0.7308, 0.6776, 0.4382 ), # 63
                ( 0.7434, 0.6877, 0.4338 ), # 64
                ( 0.7560, 0.6979, 0.4290 ), # 65
                ( 0.7686, 0.7081, 0.4241 ), # 66
                ( 0.7814, 0.7184, 0.4188 ), # 67
                ( 0.7942, 0.7288, 0.4129 ), # 68
                ( 0.8070, 0.7392, 0.4070 ), # 69
                ( 0.8200, 0.7497, 0.4007 ), # 70
                ( 0.8330, 0.7603, 0.3938 ), # 71
                ( 0.8461, 0.7710, 0.3869 ), # 72
                ( 0.8592, 0.7817, 0.3793 ), # 73
                ( 0.8725, 0.7926, 0.3712 ), # 74
                ( 0.8858, 0.8035, 0.3627 ), # 75
                ( 0.8992, 0.8145, 0.3538 ), # 76
                ( 0.9127, 0.8256, 0.3442 ), # 77
                ( 0.9262, 0.8367, 0.3340 ), # 78
                ( 0.9399, 0.8480, 0.3232 ), # 79
                ( 0.9536, 0.8593, 0.3116 ), # 80
                ( 0.9674, 0.8708, 0.2990 ), # 81
                ( 0.9814, 0.8823, 0.2856 ), # 82
                ( 0.9954, 0.8940, 0.2708 ), # 83
                ( 1.0000, 0.9057, 0.2593 ), # 84
                ( 1.0000, 0.9169, 0.2731 ), # 85
                ], N=85)
    else:
        return LinearSegmentedColormap.from_list( \
            "inferno", [NiceColour(ID="Navy"),
                        #(0.92,0.66,0.33),
                        NiceColour(ID="Orange"),
                        NiceColour(ID="Beige")], N=100)


RetroID = {"Blue":0,"Yellow":1,"Green":2,"Orange":3,"Pink":4,"Black":5}
RetroColours = [
    (44,8,238), # Blue
    (229,178,67), # Yellow
    (20,65,26), # Green
    (242,77,11), #Orange
    (210,106,162), # Pink
    (0,0,0), # Black
]

def RetroColour(k=5, ID=None, Shade=1.0):
    if not(ID is None) and ID in RetroID:
            k = RetroID[ID]
            
    CRGB = RetroColours[k%6]
    C = (CRGB[0]/255,CRGB[1]/255,CRGB[2]/255)
    return (C[0]*Shade, C[1]*Shade, C[2]*Shade)

def VisLight(E, Units="eV"):
    Scale = {'EV':1., 'HA':27.2, 'MHA':0.0272}
    if Units.upper() in Scale:
        x = Scale[Units.upper()]
    else: 
        print("Must use one of:" + ",".join(list(Scale)))
        quit()

    xGrid = {
        (2.75,3.26):(),
        (2.56,2.75):(),
        (2.48,2.56):(),
        (2.19,2.48):(),
        (2.10,2.19):(),
        (1.98,2.10):(),
        (1.65,1.98):(),
    }

    


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    SetupNiceColours()

    Txt = ['\'%s\''%(NiceColourHex[c]) for c in DarkColourID]
    print("[ %s ]"%(", ".join(Txt)))
        
    fig,(ax1,ax2,ax3) = plt.subplots(3)

    Max = 22
    SetupNiceColours()
    for k in range(Max):
        ax1.fill_between([k,k+1.],[0.,0.],[1.,1.],
                         color=NiceColour(k))

    SetupNiceColours()
    for k in range(Max):
        ax2.fill_between([k,k+1.],[0.,0.],[1.,1.],
                         color=BestColour(k))

    SetupNiceGreys()
    for k in range(Max):
        ax3.fill_between([k,k+1.],[0.,0.],[1.,1.],
                         color=NiceColour(k))

    for ax in (ax1,ax2,ax3):
        ax.set_xlim(0.,Max)
        ax.set_yticks(())
        ax.set_xticks([0.5+k for k in range(Max)])
        if ax==ax2:
            ax.set_xticklabels(list(BestColourID), rotation=90)
        else:
            ax.set_xticklabels(list(NiceColourTable), rotation=90)
    plt.tight_layout()
    plt.savefig("NiceColours-Demo.pdf")

    import numpy as np
    
    fig2, axIm = plt.subplots(1)
    x,y = np.meshgrid(np.linspace(-2,2,401),np.linspace(-2,2,401))
    r2 = x**2+y**2
    z = (1. - 3.*r2)*np.exp(-r2)
    axIm.imshow(z, vmin=-1., vmax=1., cmap=NiceCMap("coolwarm"))

    plt.show()
    
