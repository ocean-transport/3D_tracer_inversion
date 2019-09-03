import xarray as xr
import numpy as np
from scipy.linalg import pinv, eig, eigh

# Function to calculate the diffusivity tensor 

def calc_tensor(uc,vc,wc, cx,cy,cz):
    Aflux = np.array([uc, vc, wc])
    Agrad = np.array([cx, cy, cz])

    if ~(np.isnan(Agrad).any() | np.isnan(Aflux).any()):
        return -(Aflux.dot(pinv(Agrad)))
    else:
        return np.nan*(Aflux.dot(Agrad.T))
    
def calc_tensor_2D(vc,wc, cy,cz):
    Aflux = np.array([vc, wc])
    Agrad = np.array([cy, cz])

    if ~(np.isnan(Agrad).any() | np.isnan(Aflux).any()):
        return -(Aflux.dot(pinv(Agrad)))
    else:
        return np.nan*(Aflux.dot(Agrad.T))  
    

# this does the same thing as the old calc tensor; solves stacked problem,
# but does weighting but the amount of flux.
def calc_tensor_3(uc,vc,wc, cx,cy,cz):
    Afluxu = np.array([uc]).T
    Afluxv = np.array([vc]).T
    Afluxw = np.array([wc]).T
    
    Wu = np.abs(Afluxu)
    Wv = np.abs(Afluxv)
    Ww = np.abs(Afluxw)
    
    Agrad = np.array([cx, cy, cz]).T

    if ~(np.isnan(Agrad).any() |  np.isnan(1/Wu).any() |  np.isnan(1/Wv).any() |  np.isnan(1/Ww).any() | 
         np.isinf(Agrad).any() |  np.isinf(1/Wu).any() |  np.isinf(1/Wv).any() |  np.isinf(1/Ww).any()):
        
        Kx = pinv(-Agrad/Wu).dot(Afluxu/Wu)
        Ky = pinv(-Agrad/Wv).dot(Afluxv/Wv)
        Kz = pinv(-Agrad/Ww).dot(Afluxw/Ww)
    
        K = np.concatenate((Kx.T, Ky.T, Kz.T), axis=0)
        
        # return(K, Kx)
        return K
    else:
        return np.nan*np.ones((3,3), dtype='float32')
    
# Function to put tracer fluxes into individual arrays,
# and line them up along a dimension.
def get_flux_arrays(ds, list_tracers): 
    # U'C'
    testxr1 = ds['UpTRAC01p']
    testxr1['tracer_num'] = 1

    UpCp = testxr1 

    n=2
    for i in list_tracers[1:]: 
        temp = ds['Up'+i+'p']
        temp['tracer_num'] = n 
        n=n+1
        UpCp = xr.concat([UpCp, temp], dim='tracer_num')
    
    UpCp.name = 'UpCp'
    
    # V'C'
    testxr1 = ds['VpTRAC01p']
    testxr1['tracer_num'] = 1

    VpCp = testxr1 

    n=2
    for i in list_tracers[1:]: 
        temp = ds['Vp'+i+'p']
        temp['tracer_num'] = n 
        n=n+1

        VpCp = xr.concat([VpCp, temp], dim='tracer_num')

    VpCp.name = 'VpCp'
    
    # W'C'
    testxr1 = ds['WpTRAC01p']
    testxr1['tracer_num'] = 1

    WpCp = testxr1 

    n=2
    for i in list_tracers[1:]: 
        temp = ds['Wp'+i+'p']
        temp['tracer_num'] = n 
        n=n+1

        WpCp = xr.concat([WpCp, temp], dim='tracer_num')
    
    WpCp.name = 'WpCp'
    
    return [UpCp, VpCp, WpCp]

def get_flux_arrays_2D(ds, list_tracers, ftype): 
    # V'C'
    testxr1 = ds['V'+ftype+'TRAC01'+ftype]
    testxr1['tracer_num'] = 1

    VpCp = testxr1 

    n=2
    for i in list_tracers[1:]: 
        temp = ds['V'+ftype+i+ftype]
        temp['tracer_num'] = n 
        n=n+1

        VpCp = xr.concat([VpCp, temp], dim='tracer_num')

    VpCp.name = 'VpCp'
    
    # W'C'
    testxr1 = ds['W'+ftype+'TRAC01'+ftype]
    testxr1['tracer_num'] = 1

    WpCp = testxr1 

    n=2
    for i in list_tracers[1:]: 
        temp = ds['W'+ftype+i+ftype]
        temp['tracer_num'] = n 
        n=n+1

        WpCp = xr.concat([WpCp, temp], dim='tracer_num')
    
    WpCp.name = 'WpCp'
    
    return [VpCp, WpCp]

# Function to put tracer fluxes (diffusive - do to KPP and numerical) into individual arrays,
# and line them up along a dimension.
def get_diff_flux_arrays(ds, list_tracers): 
    # - K_v * dC/dz
    testxr1 = ds['DFrITr01']
    testxr1['tracer_num'] = 1

    KvCz = testxr1 

    n=2
    for i in list_tracers[1:]: 
        temp = ds['DFrI'+i]
        temp['tracer_num'] = n 
        n=n+1
        
        KvCz = xr.concat([KvCz, temp], dim='tracer_num')
    
    KvCz.name = 'KvCz'
    
    return KvCz

# function to get flux arrays for the stationary fluxes (but in 3D)
def get_stationary_flux_arrays(ds, ds_vel, list_tracers): 
    
    # anomaly from zonally avg vels
    Udagger = ds_vel.U - ds_vel.U.mean('XCcoarse')
    Vdagger = ds_vel.V - ds_vel.V.mean('XCcoarse')
    Wdagger = ds_vel.W - ds_vel.W.mean('XCcoarse')
    
    testxr_Cdagger1 = ds['TRAC01'] - ds['TRAC01'].mean('XCcoarse')
    # U'C'
    testxr1 = testxr_Cdagger1*Udagger
    testxr1['tracer_num'] = 1

    UdaggerCdagger = testxr1 
    
     # V'C'
    testxr1 = testxr_Cdagger1*Vdagger
    testxr1['tracer_num'] = 1

    VdaggerCdagger = testxr1 
    
    # W'C'
    testxr1 = testxr_Cdagger1*Wdagger
    testxr1['tracer_num'] = 1

    WdaggerCdagger = testxr1 

    n=2
    for i in list_tracers[1:]: 
        tempdagger = ds[i] - ds[i].mean('XCcoarse')
        
        temp = tempdagger*Udagger
        temp['tracer_num'] = n 
        UdaggerCdagger = xr.concat([UdaggerCdagger, temp], dim='tracer_num')
    
        temp = tempdagger*Vdagger
        temp['tracer_num'] = n 
        VdaggerCdagger = xr.concat([VdaggerCdagger, temp], dim='tracer_num')

        temp = tempdagger*Wdagger
        temp['tracer_num'] = n 

        WdaggerCdagger = xr.concat([WdaggerCdagger, temp], dim='tracer_num')
        n=n+1
        
    UdaggerCdagger.name = 'UdaggerCdagger'
    VdaggerCdagger.name = 'VdaggerCdagger'
    WdaggerCdagger.name = 'WdaggerCdagger'
    
    return [UdaggerCdagger, VdaggerCdagger, WdaggerCdagger]

# Function to put tracer grads into individual arrays,
# and line them up along a dimension.
def get_grad_arrays(ds, list_tracers): 
    # Put tracer gradients into xarrays
# dCdx
    testxr1 = ds['TRAC01_X']
    testxr1['tracer_num'] = 1

    dCdx = testxr1 

    n=2
    for i in list_tracers[1:]: 
        temp = ds[i+'_X']
        temp['tracer_num'] = n 
        n=n+1

        dCdx = xr.concat([dCdx, temp], dim='tracer_num')
    
    dCdx.name = 'dCdx'
    
    # dCdy
    testxr1 = ds['TRAC01_Y']
    testxr1['tracer_num'] = 1

    dCdy = testxr1 

    n=2
    for i in list_tracers[1:]: 
        temp = ds[i+'_Y']
        temp['tracer_num'] = n 
        n=n+1

        dCdy = xr.concat([dCdy, temp], dim='tracer_num')   
    dCdy.name = 'dCdy'
    
    # dCdz
    testxr1 = ds['TRAC01_Z']
    testxr1['tracer_num'] = 1

    dCdz = testxr1 

    n=2
    for i in list_tracers[1:]: 
        temp = ds[i+'_Z']
        temp['tracer_num'] = n 
        n=n+1

        dCdz = xr.concat([dCdz, temp], dim='tracer_num')
    dCdz.name = 'dCdz'
    
    return [dCdx, dCdy, dCdz]

def get_grad_arrays_2D(ds, list_tracers): 
    # Put tracer gradients into xarrays
    
    # dCdy
    testxr1 = ds['TRAC01_Y']
    testxr1['tracer_num'] = 1

    dCdy = testxr1 

    n=2
    for i in list_tracers[1:]: 
        temp = ds[i+'_Y']
        temp['tracer_num'] = n 
        n=n+1

        dCdy = xr.concat([dCdy, temp], dim='tracer_num')   
    dCdy.name = 'dCdy'
    
    # dCdz
    testxr1 = ds['TRAC01_Z']
    testxr1['tracer_num'] = 1

    dCdz = testxr1 

    n=2
    for i in list_tracers[1:]: 
        temp = ds[i+'_Z']
        temp['tracer_num'] = n 
        n=n+1

        dCdz = xr.concat([dCdz, temp], dim='tracer_num')
    dCdz.name = 'dCdz'
    
    return [dCdy, dCdz]

# Function to reconstruct fluxes
def flux_reconstruct(tensor, flux, grads):

    
    recUflux = -(tensor.sel(i=0, j=0)*grads.dCdx + tensor.sel(i=0, j=1)*grads.dCdy + tensor.sel(i=0, j=2)*grads.dCdz)
    recVflux = -(tensor.sel(i=1, j=0)*grads.dCdx + tensor.sel(i=1, j=1)*grads.dCdy + tensor.sel(i=1, j=2)*grads.dCdz)
    recWflux = -(tensor.sel(i=2, j=0)*grads.dCdx + tensor.sel(i=2, j=1)*grads.dCdy + tensor.sel(i=2, j=2)*grads.dCdz)

    errU = np.abs(flux.UpCp - recUflux)/np.abs(flux.UpCp)
    errV = np.abs(flux.VpCp - recVflux)/np.abs(flux.VpCp)
    errW = np.abs(flux.WpCp - recWflux)/np.abs(flux.WpCp)
    
    flux_rec=xr.Dataset({'UpCp':recUflux, 'VpCp':recVflux, 'WpCp':recWflux,
                                'errU':errU, 'errV':errV, 'errW':errW})
    
    return flux_rec

def calc_tensor_vec_constrained(uc,vc,wc, cx,cy,cz, ub,vb,wb, bx,by,bz): 
    
    bcvec = np.concatenate((uc,vc,wc))
    bbvec = np.array([0,0,0, ub, vb, wb])
    bvec = np.concatenate((bcvec,bbvec))
    
    grad = np.array([cx, cy, cz]).T
    
    E1vec = np.concatenate((grad, np.zeros_like(grad), np.zeros_like(grad)), axis=1)
    E2vec = np.concatenate((np.zeros_like(grad), grad, np.zeros_like(grad)), axis=1)
    E3vec = np.concatenate((np.zeros_like(grad), np.zeros_like(grad), grad), axis=1)
    
    Ebvec = np.array([(bx, by/2, bz/2, by/2, 0,  0,    bz/2, 0,    0),
                   (0 , bx/2, 0   , bx/2, by, bz/2, 0   , bz/2, 0),
                   (0 , 0   ,bx/2 , 0   , 0 , by/2, bx/2, by/2, bz),
                   (0 , by/2, bz/2,-by/2, 0 , 0   ,-bz/2,  0  ,  0),
                   (0 , -bx/2, 0  , bx/2 ,0 , bz/2, 0   , -bz/2, 0), 
                   (0 , 0   ,-bx/2, 0    , 0, -by/2, bx/2, by/2 ,0)])
    
    Evec = np.concatenate((E1vec, E2vec, E3vec, Ebvec), axis=0)
    
    K = -pinv(Evec).dot(bvec)
    
    K = K.reshape((3,3))
    return K