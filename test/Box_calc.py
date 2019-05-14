import numpy as np
import sys
import math
import os
import time

def safe_exp(w):
  if w <= 1:
      return np.exp(w)
  else:
      return 2.71828*w


# org_xy = np.array([[(15+x*30,16+y*32) for y in range(20)] for x in range(12)])  #set block center(x,y)
org_xy = np.array([[(x*(640.0/21),y*(360.0/13)) for x in range(1, 21)] for y in range(1, 13) ])  #set block center(x,y)
org_wh = np.array([[229.,137.],[48.,71.],[289.,245.],[185.,134.],[85.,142.],[31.,41.],[197.,191.],[237.,206.],[63.,108.]])  #set anchorsize
H0=int(12) #height
W_full=24 #width
Wgroups=8
Wpergroup=3
C_full=48 #channels in ddr
Cgroups=6
Cpergroup=8
C=45 #channels of test.txt
Confid = np.zeros([H0,W_full,9])  #make a np to save Confidence
OrgDxywh = np.zeros([H0,W_full,36]) #make a np to save dx,dy,dh,dw

#tmp = base_ddr_space[DDR_OUTPUT_FM_ADDR: DDR_OUTPUT_FM_ADDR + DDR_OUTPUT_FM_LEN * DDR_OUTPUT_FM_HEIGHT * 64].copy()

Readin = np.loadtxt('test.txt',delimiter='\t') #read txt for test
tmp =Readin.astype(np.int8) #convert to int8

Orgtemp = tmp.reshape(H0, Wpergroup, Cgroups, Wgroups, Cpergroup)  # cnvert input to a suitable np  ,C=45
OrgOuttmp = Orgtemp.transpose(0, 3, 1, 2, 4)
OrgOut = OrgOuttmp.reshape(H0, W_full, C_full)
Confid = OrgOut[:, 0:20, 0:9]  # set confidence np
OrgDxywh = OrgOut[:, :, 9:45]  # set (dx,dy,dh,dw) np
Dxywh = OrgDxywh.reshape(H0, W_full, 9, 4)  # use Dxywh to represent the (dx,dy,dh,dw) of each anchor

# maxindex = np.unravel_index(Confid.argmax(), Confid.shape)
# [Hindex,Windex,anchorindex] = maxindex   #get the maxindex of the max confidence in Congfid
maxindex = Confid.flatten().argsort()[-1]

Hindex = maxindex // (20 * 9)
maxindex -= Hindex * 20 * 9
Windex = maxindex // 9
anchorindex = maxindex - Windex * 9
# print(maxindex,Hindex,Windex,anchorindex)
[Xhwk, Yhwk] = org_xy[Hindex, Windex]  # get the center (x,y) of the block weget according to the maxindex
[Whwk, Hhwk] = org_wh[anchorindex]  # get the anchor-size of the anchor we get according to the maxindex
[dx, dy, dw, dh] = Dxywh[Hindex, Windex, anchorindex,] / 4  # Divided by a coefficient"4" to get (dx,dy,dw,dh)

X = Xhwk + Whwk * dx
Y = Yhwk + Hhwk * dy  # the center(x,y) of the baouning box
W = Whwk * safe_exp(dw)  # the width of the bounding box
H = Hhwk * safe_exp(dh)  # the hight of the bounding box

Xmin = int(X - W / 2)  # Min of the horizontal axis
Xmax = int(X + W / 2 + 1)  # Max of the horizontal axis
Ymin = int(Y - H / 2)  # Min of the vertical axis
Ymax = int(Y + H / 2 + 1)  # Max of the vertical axis（output should be int)
if Xmin < 0:
    Xmin = 0
if Xmax > 639:
    Xmax = 639
if Ymin < 0:
    Ymin = 0
if Ymax > 359:
    Ymax = 359
# timer stop after PS has received image

print(Xmin,Xmax,Ymin,Ymax)     ####输出计算结果