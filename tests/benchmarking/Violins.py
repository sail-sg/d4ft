import numpy as np
import matplotlib.pyplot as plt

from NiceColours import *
# from ReadSingletSet import *


def ToSpread(Data, dy=0.1):
    Range = np.max(Data)-np.min(Data)
    N = (Range/dy+6)
    
    yp = np.linspace(np.min(Data)-3.*dy, np.max(Data)+3.*dy, 2*int(N))
    wp = 0.*yp

    for D in Data:
        wp += np.exp(-0.5*(yp-D)**2/dy**2)/np.sqrt(2.*np.pi*dy**2)

    return yp, wp
    
    
def AddViolin(ax,
              X, Data,
              dy=0.1,
              width=0.5,
              color_list = None,
              Mask=None,
              MAEMode="H", MAEWidth=0.6,
):
    Max_wp = 0.
    for D in Data:
        yp,wp = ToSpread(D, dy)
        Max_wp = max(Max_wp, wp.max())

    if width>0.:
        xScale = width/Max_wp
    else:
        xScale = -width

    if color_list is None \
       or not(len(color_list)==len(X)):
        color_list = range(len(X))
    
    for K,x in enumerate(X):
        if not Mask is None:
            if not(Mask[K]):
                continue

            
        yp,wp = ToSpread(Data[K],dy)
        wp *= xScale

        MSE = np.mean(Data[K])
        MAE = np.mean(np.abs(Data[K]))

        cc = NiceColour(color_list[K])
        
        ax.fill_betweenx(yp, x-wp, x+wp,
                         color=cc)
        YMAE = np.sign(MSE)*MAE
        if MAEMode.upper()=="V":
            ax.plot([x,x],[0.,YMAE],"-",
                    linewidth=1, color=(0,0,0))
        else:
            ax.plot([x+MAEWidth/2.,x-MAEWidth/2.],[YMAE,YMAE],"-",
                    linewidth=1, color=(0,0,0))

# if False:
#     Basis="aug-cc-pvtz"
#     Methods, Systems, AllData \
#         = ReadSingletSet("lowest_singlet_%s.txt"%(Basis))
# 
#     Methods = Methods[:-1] # Remove 'Ref'
# 
#     X = np.arange(len(Methods))
#     Data = [None]*len(Methods)
#     for K,M in enumerate(Methods):
#         Data[K] = [AllData[S][M]-AllData[S]['Ref'] for S in Systems]
# 
#     fig, ax = plt.subplots(1, figsize=(6,3))
#     AddViolin(ax, X, Data, dy=0.2)
# 
#     ax.plot([-20,20],[0,0],":k", linewidth=1) # Add zero line
# 
#     ax.set_xticks(X)
#     ax.set_xticklabels([M.upper() for M in Methods], rotation=45)
# 
#     ax.axis([X.min()-0.5, X.max()+0.5,-3.,12.])
# 
#     plt.tight_layout()


# if True:
#     Systems, Data = ReadMultiSet()
#     # print(Data)
#     Tags = list(Data['6-31G']['pbe'])
#     print(Tags)
# 
#     ColList = ["Blue", "Red", "Teal", "Brown"]
# 
#     def NiceDFA(DFA):
#         NiceDFAList = {}
#         X = DFA.upper()
#         if not(X in NiceDFAList): return X
#         else: return NiceDFAList[X]
# 
#     fig, axs = plt.subplots(2, figsize=(6,5))
#     for Kax,Basis in enumerate(list(Data)):
#         ax = axs[Kax]
# 
#         AddBorder(
#             ax.text(0.01,0.03, Basis,
#                     fontsize=14,
#                     transform = ax.transAxes,
#             )
#         )
#         for KQ,Q in enumerate(Tags):
#             AddBorder(
#                 ax.text(0.35 + KQ*0.15,0.9,Q,
#                         color=ColList[KQ],
#                         fontsize=14,
#                         horizontalalignment="center",
#                         transform=ax.transAxes,
#                 )
#             )
# 
#         XT, XTL = [], []
#         for KDFA, DFA in \
#             enumerate( ('hf','pbe','pbe0','pbe50', 'blyp','b3lyp','bhhlyp') ):
#             X = np.arange(4) + KDFA*4.5
#             VData = np.array([Data[Basis][DFA][i] for i in Tags])
# 
#             VMeanCut = 3.5
#             VMean = VData.mean(axis=1)
#             VMA = np.abs(VMean)
#             for P,M in enumerate(VMean):
#                 if np.abs(M)>VMeanCut:
#                     AddBorder(
#                         ax.text(X[P],-2.,"ME: %+.2f"%(M),
#                                 rotation=90,
#                                 fontsize=14,
#                                 verticalalignment="bottom",
#                                 horizontalalignment="center",
#                                 color=ColList[P]
#                         )
#                     )
#             
#             AddViolin(ax, X, VData, dy=0.25,
# #                       Mask=(VMA<VMeanCut),
#                       color_list=ColList)
# 
#             XT += [np.mean(X),]
#             XTL += [NiceDFA(DFA),]
#             
#         ax.plot([-10,50],[0,0],":k", linewidth=1) # Add zero line
# 
#         ax.set_ylabel("Error in gap [eV]", fontsize=12)
#         ax.set_xticks(XT)
#         ax.set_xticklabels(XTL)
#         ax.set_yticks([-4,-2,0,2,4])
#         ax.axis([np.min(XT)-1.7, np.max(XT)+1.7,-5.,5.])
# 
#     plt.tight_layout()
#     plt.savefig("FigCompare.pdf")
#     plt.savefig("FigCompare.eps")

