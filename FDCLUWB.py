# -*- coding: utf-8 -*-

from simfunc import Locpred,getpos,rotx
import numpy as np
import matplotlib.pyplot as plt

def simplot(xk,xg,robnum):
    xgs = rotx(xg,0)
    posc = np.array([getpos(xk[i]) for i in range(robnum)])
    posg = np.array([getpos(xgs[i]) for i in range(robnum)])
    [plt.scatter(posc[i][:,0],posc[i][:,1],marker='*') for i in range(robnum)]
    [[plt.text(posc[r,i,0],posc[r,i,1],str(i)) for i in range(3)] for r in range(robnum)]
    [plt.scatter(posg[i][:,0],posg[i][:,1],marker='o',alpha=0.5) for i in range(robnum)]
    [[plt.text(posg[r,i,0],posg[r,i,1],str(i)) for i in range(3)] for r in range(robnum)]
    plt.axis('equal')
    
def rotxk(xk,xg,n):
    xk = rotx(xk,n)
    a = xg[0,2]
    R = np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]]) #get rotation matrix
    xk[:,0:2] = np.dot(xk[:,0:2],R.T)       #rotation
    xk[:,0:2] = xk[:,0:2] + xg[0,0:2]      #translation
    xk[:,2] = xk[:,2]+xg[0,2]
    return xk

robnum = 5#num of robots
#locpred = Locpred(robnum,50) 
ll = [Locpred(robnum,30) for i in range(robnum)]
xg = np.random.randn(robnum,3)*2000

thet = np.linspace(0,np.pi*2-0.1,301)
#x = 15000*np.cos(thet)
x = 15000*thet/2+15000
y = 15000*np.sin(thet)
target = np.array([x,y]).T

xka = np.zeros([301,robnum,robnum,3])
xga = np.zeros([301,robnum,3])

obs = np.array([[16000,5000],[37000,0],[50000,-14000]])
#plt.plot(target[:,0],target[:,1])
#plt.scatter(obs[:,0],obs[:,1])
#plt.axis('equal')
#%%
d = 2000

err = []
#plt.figure(2)
for t in range(301):
    xt = target[t,:]
    xk = np.array([ll[i].update(xg) for i in range(robnum)])
    for n in range(robnum):
        xk[n] = rotxk(xk[n],xg,0)
    for n in range(robnum):
        dis = np.linalg.norm(xk[n,:,0:2]-xk[n,n,0:2],axis=1).reshape(-1,1)
        dis[dis==0] = d
        fr = 1/d-1/dis
        fr[fr>0]=0
        fr = (xk[n,:,0:2]-xk[n,n,0:2])/dis*fr
        xg[n,0:2] = xg[n,0:2]+200000*fr.sum(axis=0)
        
        #obs = np.vstack([xt,obs])
        dis = np.linalg.norm(obs-xk[n,n,0:2],axis=1).reshape(-1,1)
        dis[dis==0] = d
        fr = 1/d-1/dis
        fr[fr>0]=0
        fr = (obs-xk[n,n,0:2])/dis*fr
        xg[n,0:2] = xg[n,0:2]+250000*fr.sum(axis=0)
        
        fa = (xk[n][n,0:2]-xt)/np.linalg.norm(xk[n][n,0:2]-xt)
        xg[n,0:2] = xg[n,0:2]-280*fa
        
        a = xk[n,n,2]
        xa = xt.copy()
        xa = xa-xg[n,0:2]
        R = np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
        xa = np.dot(xa,R)
        a = np.arctan2(xa[1],xa[1])
        xg[n,2] = xg[n,2]+0.05*a

    xga[t,:,:] = xg
    xka[t,:,:,:] = xk

    posg = np.array([getpos(xg[i]) for i in range(robnum)])
    posc = np.array([getpos(xk[n][i]) for i in range(robnum)])
    err.append(abs((posg - posc).mean()))
    
    plt.cla()
    
    [plt.scatter(posg[i][:,0],posg[i][:,1],marker='o',alpha=0.5) for i in range(robnum)]
    [plt.scatter(posc[i][:,0],posc[i][:,1],marker='*') for i in range(robnum)]
    plt.plot(target[0:t,0],target[0:t,1],marker='.')
    plt.scatter(obs[:,0],obs[:,1],s=200)
    
    
    '''
    plt.scatter(xg[:,0],xg[:,1],marker='+')
    plt.scatter(xg[:,0]+400*np.cos(xg[:,2]),xg[:,1]+400*np.sin(xg[:,2]),marker='+')
    plt.axis('equal')
    '''
    plt.pause(0.001)
    
