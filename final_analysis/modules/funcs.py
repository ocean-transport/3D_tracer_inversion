import numpy as np
import xarray as xr
import sys
sys.path.append('../../m2lines/channel-coarse-grain-pipeline/modules/')
import filter_coarsen_func as fcf # modu

def spatial_filter(da, Lfilter, dims=['XC', 'YC']): 
    dx = 5e3 # model resolution, can be tweaked
    sigma = Lfilter/dx/np.sqrt(12)
    
    return fcf.apply_gauss(da, sigma, dims=dims)


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

def get_grad_arrays(ds, list_tracers): 
    # Put tracer gradients into xarrays
# dCdx
    testxr1 = ds['dTRAC01dx']
    testxr1['tracer_num'] = 1

    dCdx = testxr1 

    n=2
    for i in list_tracers[1:]: 
        temp = ds['d'+i+'dx']
        temp['tracer_num'] = n 
        n=n+1

        dCdx = xr.concat([dCdx, temp], dim='tracer_num')
    
    dCdx.name = 'dCdx'
    
    # dCdy
    testxr1 = ds['dTRAC01dy']
    testxr1['tracer_num'] = 1

    dCdy = testxr1 

    n=2
    for i in list_tracers[1:]: 
        temp = ds['d'+i+'dy']
        temp['tracer_num'] = n 
        n=n+1

        dCdy = xr.concat([dCdy, temp], dim='tracer_num')   
    dCdy.name = 'dCdy'
    
    # dCdz
    testxr1 = ds['dTRAC01dz']
    testxr1['tracer_num'] = 1

    dCdz = testxr1 

    n=2
    for i in list_tracers[1:]: 
        temp = ds['d'+i+'dz']
        temp['tracer_num'] = n 
        n=n+1

        dCdz = xr.concat([dCdz, temp], dim='tracer_num')
    dCdz.name = 'dCdz'
    
    return [dCdx, dCdy, dCdz]