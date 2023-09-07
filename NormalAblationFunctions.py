# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 15:02:30 2023

@author: Martin Chaigne
"""

import numpy as np
from scipy.spatial import distance_matrix
from matplotlib import tri

se=0.999999
def erod(xx,zz,l):
    #points obtained by moving by a distance l along the normal
    gradvec=np.array([np.gradient(zz),-np.gradient(xx)])#computation of a normal vector
    vecn=l*gradvec/np.linalg.norm(gradvec,axis=0)#normal vector of norm l
    xxprop=xx+vecn[0,:]
    zzprop=zz+vecn[1,:]#new points obtained     
    #keep only points which are at a distance farther than l from the initial curve
    new_coord=np.transpose(np.array([xxprop,zzprop]))
    ex_coord=np.transpose(np.array([xx,zz]))
    cleaned_new_coord=np.delete(new_coord,np.unique(np.asarray((distance_matrix(new_coord,ex_coord)<l*se).nonzero()[0])),axis=0)
    final_xx,final_zz=cleaned_new_coord[:,0],cleaned_new_coord[:,1]
    return final_xx, final_zz

def distance_neighbours(ex,new,n):
    distmat=np.zeros((2*n+1,np.shape(new)[1]))
    for k in range(-n,n+1):
        ex_rollk=np.roll(ex,k)
        distmat[n+k]=np.sqrt(np.sum((new-ex_rollk)**2,axis=0))
    return np.concatenate((distmat[:n],distmat[n+1:]),axis=0)   

def erod_improved(xx,zz,l,n):
    #points obtain by moving by a distance l along the normal
    gradvec=np.array([np.gradient(zz),-np.gradient(xx)])# Calcul du vecteur normal initial.
    vecn=l*gradvec/np.linalg.norm(gradvec,axis=0)
    xxprop=xx+vecn[0,:]
    zzprop=zz+vecn[1,:]    
    #keep only points which are at a distance farther than l from the initial curve
    new_coord=np.array([xxprop,zzprop])
    ex_coord=np.array([xx,zz])
    cleaned_new_coord=np.delete(new_coord,np.unique((distance_neighbours(ex_coord,new_coord,n)<l*se).nonzero()[1]),axis=1)
    final_xx,final_zz=cleaned_new_coord[0,:],cleaned_new_coord[1,:]
    return final_xx, final_zz

def erod_non_uniform_improved(xx,zz,l,f_er,pp,n):
    ff=f_er(xx,zz,pp)
    #points obtain by moving by a distance l along the normal
    gradvec=np.array([np.gradient(zz),-np.gradient(xx)])
    vecn=l*ff*gradvec/np.linalg.norm(gradvec,axis=0)
    xxprop=xx+vecn[0,:]
    zzprop=zz+vecn[1,:]    
    #keep only points which are at a distance farther than l from the initial curve
    new_coord=np.array([xxprop,zzprop])
    ex_coord=np.array([xx,zz])
    f_er_mat=np.array([np.concatenate((np.roll(ff,n-i)[:n],np.roll(ff,n-i)[n+1:2*n+1])) for i in range(len(xxprop))]).transpose()
    cleaned_new_coord=np.delete(new_coord,np.unique((distance_neighbours(ex_coord,new_coord,n)<l*se*f_er_mat).nonzero()[1]),axis=1)
    final_xx,final_zz=cleaned_new_coord[0,:],cleaned_new_coord[1,:]
    return final_xx, final_zz

def distance_neighbours_2d(ex,new,dx,dy,nx,ny):
    distmat=np.zeros((2*nx+1,2*ny+1,len(ex)))
    for k in range(-nx,nx+1):
        for j in range(-ny,ny+1):
            ex_roll_kj=np.roll(ex,-dx*j-k,axis=0)
            distmat[nx+k,ny+j]=np.sqrt(np.sum((new-ex_roll_kj)**2,axis=1))
    return distmat

def propag_neighbours_2d(ff,lx,ly,nx,ny):
    neighbmat=np.zeros((2*nx+1,2*ny+1,len(ff)))
    for k in range(-nx,nx+1):
        for j in range(-ny,ny+1):
            ff_roll_kj=np.roll(ff,-lx*j-k)
            neighbmat[nx+k,ny+j]=ff_roll_kj
    return neighbmat   

def erod_improved_2d(xx,yy,zz,nx,ny,l):
    #initialization
    dx,dy=len(xx),len(yy)
    XX,YY=np.meshgrid(xx,yy)
    #points obtain by moving by a distance l along the normal
    gradvec=np.append(np.gradient(zz.transpose(),xx,yy),[0*zz.transpose()-1],axis=0)#normal vector
    vecn=l*gradvec/np.linalg.norm(gradvec,axis=0)
    xxprop=XX+vecn[0,:].transpose()
    yyprop=YY+vecn[1,:].transpose()
    zzprop=zz+vecn[2,:].transpose()
    #keep only points which are at a distance farther than l from the initial curve
    new_coord=np.transpose(np.array([xxprop.flatten(),yyprop.flatten(),zzprop.flatten()]))
    ex_coord=np.transpose(np.array([XX.flatten(),YY.flatten(),zz.flatten()]))
    cleaned_new_coord=np.delete(new_coord,np.unique((distance_neighbours_2d(ex_coord,new_coord,dx,dy,nx,ny)<l*se).nonzero()[2]),axis=0)
    #return points on a regular grid by interpolating
    interpolator = tri.LinearTriInterpolator(tri.Triangulation(cleaned_new_coord[:,0],cleaned_new_coord[:,1]), cleaned_new_coord[:,2])
    return interpolator(XX, YY)

def erod_nonuniform_improved_2d(xx,yy,zz,nx,ny,l,fer,pp):
    #initialization
    lx,ly=len(xx),len(yy)
    XX,YY=np.meshgrid(xx,yy)
    #points obtain by moving by a distance l along the normal
    gradvec=np.append(np.gradient(zz.transpose(),xx,yy),[0*zz.transpose()-1],axis=0)#normal vector 
    vecn=l*fer(XX,YY,zz,pp).transpose()*gradvec/np.linalg.norm(gradvec,axis=0)
    xxprop=XX+vecn[0,:].transpose()
    yyprop=YY+vecn[1,:].transpose()
    zzprop=zz+vecn[2,:].transpose()
    #keep only points which are at a distance farther than l from the initial curve
    new_coord=np.transpose(np.array([xxprop.flatten(),yyprop.flatten(),zzprop.flatten()]))
    ex_coord=np.transpose(np.array([XX.flatten(),YY.flatten(),zz.flatten()]))
    ff=fer(XX,YY,zz,pp).flatten()
    cleaned_new_coord=np.delete(new_coord,np.unique((distance_neighbours_2d(ex_coord,new_coord,lx,ly,nx,ny)<l*se*propag_neighbours_2d(ff,lx,ly,nx,ny)).nonzero()[2]),axis=0)
    #return points on a regular grid by interpolating
    interpolator = tri.LinearTriInterpolator(tri.Triangulation(cleaned_new_coord[:,0],cleaned_new_coord[:,1]), cleaned_new_coord[:,2])
    return interpolator(XX, YY)