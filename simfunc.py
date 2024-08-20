# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial.distance import pdist


def getpos(cord):
    l= 300
    x,y,t = cord
    pos = np.array([[x,y],[x+l*np.cos(t+np.pi/6),y+l*np.sin(t+np.pi/6)],[x+l*np.cos(t-np.pi/6),y+l*np.sin(t-np.pi/6)]])
    return pos

def getdis(pos0,posg):
    dist = pdist(np.vstack([pos0,posg]))*[0,0,1,1,1,0,1,1,1,1,1,1,0,0,0]
    dist = dist[dist>0]
    return dist

def rotx(xs,n):
    x = xs.copy()
    a = x[n,2]
    x[:,0:2] = x[:,0:2] - x[n,0:2]      #translation
    x[:,2] = x[:,2]-a
    R = np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]]) #get rotation matrix
    x[:,0:2] = np.dot(x[:,0:2],R)       #rotation
    return x



def getforce(x0,x1,dist):    #calculate force from pos0 to pos1
    pos0 = getpos(x0)
    pos1 = getpos(x1)
    cx,cy = pos1.mean(axis=0)    #get guess center of robot
    dis0 = getdis(pos0,pos1)     #get guess distance
    f = (dist-dis0)*0.01         #get optimize force
    d = np.array([[pos1[i]-pos0[j] for i in range(3)] for j in range(3)]).reshape(9,2) #get x y distance among uwb
    beta = np.array([np.arctan2(d[i,1],d[i,0]) for i in range(9)])  #get force direction
    alpha = np.arctan2(pos1[:,1]-cy,pos1[:,0]-cx)          #get robot center to uwb direction
    alpha = np.array([alpha,alpha,alpha]).reshape(-1)
    gamma = beta-alpha                        #get angle between force direction and center direction
    wr = sum(f*np.sin(gamma)*0.02)            #get rotation force 
    vx = sum(f*np.cos(gamma)*np.cos(alpha))   #get translation x force 
    vy = sum(f*np.cos(gamma)*np.sin(alpha))   #get translation y force 
    return np.array([vx,vy,wr])

def fr(xk,n,distmat):
    dd = distmat[[0,n],:][:,[0,n],:]
    for t in range(300):
        forcmat = np.zeros([2,2,3])
        for i in range(2):
            for j in range(2):
                if i!=j:
                    forcmat[i,j] =  getforce(xk[j],xk[i],dd[j][i]) #calculate force from j to i
        fsum = forcmat.sum(axis=1)
        if (abs(fsum).sum())<5:
            break
        xk = xk+fsum
        xk = rotx(xk,0)
    return xk

def frall(xk,distmat,robnum):
    for t in range(200):
        forcmat = np.zeros([robnum,robnum,3])
        for i in range(robnum):
            for j in range(robnum):
                if i!=j:
                    forcmat[i,j] =  getforce(xk[j],xk[i],distmat[j][i]) #calculate force from j to i
        fsum = forcmat.sum(axis=1)
        if (abs(fsum).mean())<3:
            break
        xk = xk+fsum
        xk = rotx(xk,0)
    return xk


class Locpred():
    def __init__(self,robnum,noise):
        self.robnum = robnum
        self.noise = noise
        self.distmat = np.zeros([robnum,robnum,9])
        self.xk = np.random.randn(robnum*3).reshape(robnum,3)*2000
    def update(self,xg):
        self.xg = xg.copy()
        self.xg = rotx(self.xg,0)
        posg = np.array([getpos(xg[i]) for i in range(self.robnum)])
        for i in range(self.robnum):
            for j in range(self.robnum):
                if i!=j:
                    self.distmat[i,j,:] = getdis(posg[i],posg[j])+np.random.randn(9)*self.noise
        for n in range(1,self.robnum):
            self.xk[[0,n],:] = fr(self.xk[[0,n],:],n,self.distmat)
        self.xk = frall(self.xk,self.distmat,self.robnum)
        return self.xk
    

