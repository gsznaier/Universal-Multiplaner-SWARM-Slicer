import numpy as np
import datetime
from itertools import permutations, combinations
import struct
from slicer.utils import general
import matplotlib.pyplot as plt

#############################################
#   Tube Dynamics and Collision Checking    #
#############################################

def compute_minimum_control_effort(coeff, l_bound=0, u_bound=90,eps=1e-5):
    """
    This function computes the minimum control effort needed to satisfy the given
    coefficents
        coeff:      1x4 array of coefficients for a 3rd order polynomial
        l_bound:    scalar value denoting minimum allowed control effort 
        u_bound:    scalar value denoting maximum allowed control effort
        
    """
    # this provides a set of possible roots, which may be imaginary or beyond specified control bounds
    if len(coeff.shape) > 1:
        roots_vals = np.apply_along_axis(func1d=np.roots, axis=1, arr=coeff)
    else:
        roots_vals = np.roots(coeff)
        
    roots_vals = np.real_if_close(np.around(roots_vals, 6))

    effort_mask = (roots_vals >= l_bound) & (roots_vals <= u_bound) & ~np.iscomplex(roots_vals)

    valid_roots = np.where(effort_mask == False, np.inf, roots_vals)
    inds = np.where(np.all(effort_mask == False,axis=-1), 0, np.argmin(np.abs(valid_roots),axis=-1))
    
    if len(coeff.shape) > 1:
        valid_roots = valid_roots[np.arange(coeff.shape[0]),inds]
    else:
        valid_roots = valid_roots[inds]
    
    return valid_roots
   
def compute_val_from_control_effort(coeff, u):
    """
    This function computes the LOCAL r or z tip value based on provided
    polynomial coefficients and control effort.
        coeff:  1x4 array of coefficients for a 3rd order polynomial
        polyval format: p[0]*x**(N-1) + p[1]*x**(N-2) + ... + p[N-2]*x + p[N-1]
        u:      scalar minimum control effort
    """
    if np.iscomplexobj(u):
        if np.all(np.isreal(u)):
            u = u.real
        else:
            raise ValueError(f"u value {u} is complex!")

    val = np.polyval(coeff,u)

    val = np.real_if_close(np.around(val, 6))

    
    return np.array(val)

def compute_z_tip_from_r_tip(r_coeff, z_coeff, r_tip, l_bound=0, u_bound=90):
    """
    This function computes the corresponding LOCAL z tip value based on the
    provided polynomial coefficients, the current LOCAL r tip value and the
    allowed control effort bounds
        r_coeff:    1x4 array of coefficients for a 3rd order polynomial
                    for the tip radius function
                    
        z_coeff:    1x4 array of coefficients for a 3rd order polynomial
                    for the tip z depth function
                    
        l_bound:    scalar value denoting minimum allowed control effort 
        u_bound:    scalar value denoting maximum allowed control effort
    """
    
    if isinstance(r_tip,np.ndarray) and r_tip.ndim > 0:
        shifted_r_coeff = r_coeff.copy()*np.ones((r_tip.shape[0],r_coeff.shape[0]))
        shifted_r_coeff[...,-1] = r_coeff[...,-1] - r_tip
        min_u = compute_minimum_control_effort(shifted_r_coeff, l_bound=l_bound, u_bound=u_bound)
        corrected_min_u = np.where(np.isinf(min_u), u_bound, min_u)
        z_tip = compute_val_from_control_effort(z_coeff, corrected_min_u)

    else:
        shifted_r_coeff = r_coeff.copy()
        shifted_r_coeff[-1] = r_coeff[-1] - r_tip
        min_u = compute_minimum_control_effort(shifted_r_coeff, l_bound=l_bound, u_bound=u_bound)
        corrected_min_u = np.where(np.isinf(min_u), u_bound, min_u)
        z_tip = compute_val_from_control_effort(z_coeff, corrected_min_u)
    
    return z_tip

def compute_r_tip_from_z_tip(r_coeff, z_coeff, z_tip, l_bound=0, u_bound=90):
    """
    This function computes the corresponding LOCAL z tip value based on the
    provided polynomial coefficients, the current LOCAL r tip value and the
    allowed control effort bounds
        r_coeff:    1x4 array of coefficients for a 3rd order polynomial
                    for the tip radius function
                    
        z_coeff:    1x4 array of coefficients for a 3rd order polynomial
                    for the tip z depth function
                    
        l_bound:    scalar value denoting minimum allowed control effort 
        u_bound:    scalar value denoting maximum allowed control effort
    """
    
    if isinstance(z_tip,np.ndarray) and z_tip.ndim > 0:
        shifted_z_coeff = z_coeff.copy()*np.ones((z_tip.shape[0],z_coeff.shape[0]))
        shifted_z_coeff[...,-1] = z_coeff[...,-1] - z_tip
        min_u = compute_minimum_control_effort(shifted_z_coeff, l_bound=l_bound, u_bound=u_bound)
        corrected_min_u = np.where(np.isinf(min_u), u_bound, min_u)
        r_tip = compute_val_from_control_effort(r_coeff, corrected_min_u)
    else:
        shifted_z_coeff = z_coeff.copy()
        shifted_z_coeff[-1] = z_coeff[-1] - z_tip
        min_u = compute_minimum_control_effort(shifted_z_coeff, l_bound=l_bound, u_bound=u_bound)
        corrected_min_u = np.where(np.isinf(min_u), u_bound, min_u)
        r_tip = compute_val_from_control_effort(r_coeff, corrected_min_u)

    return r_tip

def compute_z_translation_correction(z_goal, z):
    """
    This function computes the outer tube displacement in the local frame needed
    to hit the desired target point.
        z_goal: The target z depth we want to reach given a target path point
        z:      The generated z based on the tube dynamics with a given r_tip
    """
    
    displacement = z_goal - z
    
    return displacement

def compute_range_of_allowed_control_efforts(printer, 
                                             r_lower_range=-1000, r_upper_range=1000,
                                             z_lower_range=-1000, z_upper_range=1000,
                                             min_u_bound=-1000, max_u_bound=1000,
                                             resolution=0.01,
                                             ):
    
    # coefficients for r and z polynomials
    r_coeffs = printer["coeffs"][0]
    z_coeffs = printer["coeffs"][1]
    
    #initial max and min u bound:
    min_u = min_u_bound
    max_u = max_u_bound
    
    # find the maximum and minimum possible u bounds for r
    possible_rs, r_possible_control_efforts = [], []
    for r in np.linspace(r_lower_range,r_upper_range, num=int(np.ceil(((r_upper_range-r_lower_range)/resolution)))):
        coeffs = np.asarray(r_coeffs).copy()
        coeffs[-1] = coeffs[-1]-r
        u = compute_minimum_control_effort(coeffs, l_bound=min_u_bound, u_bound=max_u_bound,eps=1e-5)

        if not np.isinf(u) and np.isreal(u):
            possible_rs.append(r)
            if np.iscomplexobj(u):
               r_possible_control_efforts.append(u.real)
            else:
               r_possible_control_efforts.append(u)

    min_u = np.maximum(min_u, np.min(r_possible_control_efforts))
    max_u = np.minimum(max_u, np.max(r_possible_control_efforts))
    
    
    # find the maximum and minimum possible u bounds for z
    possible_zs, z_possible_control_efforts = [], []   
    for z in np.linspace(z_lower_range,z_upper_range, num=int(np.ceil(((z_upper_range-z_lower_range)/resolution)))):
        coeffs = np.asarray(z_coeffs).copy()
        coeffs[-1] = coeffs[-1]-z
        u = compute_minimum_control_effort(coeffs, l_bound=min_u_bound, u_bound=max_u_bound,eps=1e-5)

        if not np.isinf(u) and np.isreal(u):
            possible_zs.append(z)
            if np.iscomplexobj(u):
               z_possible_control_efforts.append(u.real)
            else:
               z_possible_control_efforts.append(u)
            
    
    # update bounds
    min_u = np.maximum(min_u, np.min(z_possible_control_efforts))
    max_u = np.minimum(max_u, np.max(z_possible_control_efforts))
    
    return min_u, max_u
     
def compute_local_occupied_region(r_coeff, z_coeff, min_height, max_height, tip_p, l_bound=0, u_bound=90, resolution=0.1):
    """
    This function samples along the tube spine and computes the points in local space
    that are occupied by the printer CRT.
        r_coeff:    1x4 array of coefficients for a 3rd order polynomial
                    for the tip radius function
                    
        z_coeff:    1x4 array of coefficients for a 3rd order polynomial
                    for the tip z depth function
        min_height: scalar value that specifies the minimum allowed height of the CRT
        max_height: scalar value that specifies the maximum allowed height of the CRT
        tip_p:      1x3 vector that contains the tip location in local cylindrical coordinates
        l_bound:    scalar value denoting minimum allowed control effort 
        u_bound:    scalar value denoting maximum allowed control effort
        resolution: the sampling frequency to be used
    """
    min_radius = compute_val_from_control_effort(r_coeff, u_bound)
    rs = np.append(np.arange(min_radius, tip_p[0], resolution), tip_p[0])
    zs = compute_z_tip_from_r_tip(r_coeff, z_coeff, rs, l_bound=l_bound, u_bound=u_bound)
    
    reachable = check_if_reachable(r_coeff, z_coeff, min_height, max_height, tip_p, l_bound=l_bound, u_bound=u_bound)
    if np.abs(tip_p[2]) > np.abs(zs[-1]):
        displacement = -1*(np.abs(tip_p[2]) - np.abs(zs[-1]))
        vzs = -1*np.arange(0, np.abs(zs[0]+displacement), resolution)
        vrs = rs[0]*np.ones(len(vzs))
        rs = np.concatenate((vrs, rs))
        zs = np.concatenate((vzs, zs+displacement))
        
    ts = tip_p[1]*np.ones(len(rs))
        
    return np.stack((rs,ts,-1*zs),axis=-1), reachable
    
def check_if_reachable(r_coeff, z_coeff, min_height, max_height, tip_p, l_bound=0, u_bound=90):
    """
    This function checks if the provided point is reachable given the control
    effort bounds and the specified printer CRT coefficients.
        r_coeff:    1x4 array of coefficients for a 3rd order polynomial
                    for the tip radius function
                    
        z_coeff:    1x4 array of coefficients for a 3rd order polynomial
                    for the tip z depth function
        
        min_height: scalar value that specifies minimum amount that tube 
                    can translate towards origin
        max_height: scalar value that specifies how far outer tube can 
                    translate 
                    
        tip_p:      1x3 vector that contains the tip location in local 
                    cylindrical coordinates
                    
        l_bound:    scalar value denoting minimum allowed control effort 
        u_bound:    scalar value denoting maximum allowed control effort

    """
    
    # determine if we are checking only one tip or multiple tips at once
    if len(tip_p.shape) > 1:
        mask = np.zeros(tip_p.shape[0])
        shifted_r_coeff = r_coeff.copy()*np.ones((tip_p.shape[0],len(r_coeff)))
    else:
        mask = np.zeros(0)
        shifted_r_coeff = r_coeff.copy()

    
    shifted_r_coeff[...,-1] -= tip_p[...,0]
    r_u = compute_minimum_control_effort(shifted_r_coeff, l_bound=l_bound, u_bound=u_bound,eps=1e-5)

    mask = np.logical_not(np.logical_and(np.isinf(r_u), np.isreal(r_u)))

    pred_z = -1*compute_z_tip_from_r_tip(r_coeff, z_coeff, tip_p[...,0], l_bound=l_bound, u_bound=u_bound)

    mask = np.where((tip_p[...,2] - pred_z) < 0, False, mask)
    mask = np.where((tip_p[...,2] - pred_z) > max_height, False, mask)
    mask = np.where(pred_z < min_height, False, mask)
    
    return mask

def return_reachable_pts(coeffs, min_r, max_r, min_height, max_height, local_pnts, l_bound=0, u_bound=90):
    
    cr,ca,cz = general.calcCylindCord(local_pnts)
    cylind_local_pts = np.stack((cr,ca,cz),axis=-1)

    # prevent going beyond radial limits
    clip_r = np.where(cr >= max_r, max_r, cr)             # prevent R from being too large
    clip_r = np.where(clip_r <= min_r, min_r, clip_r) # prevent R from being negative/too small

    
    # shift based on allowed outer translation limit
    cz_shifted = np.where(cz-coeffs[2] >= 0, max_height-coeffs[2], cz)      # upper limit
    cz_shifted = np.where(cz_shifted <= min_height, min_height, cz_shifted) # lower limit

    computed_r = compute_r_tip_from_z_tip(coeffs[0], coeffs[1], -1*cz_shifted, l_bound=l_bound, u_bound=u_bound)
    
    
    reachable_r = np.minimum(clip_r, computed_r)
    reachable_z = np.minimum(cz, max_height)
    reachable_z = np.maximum(reachable_z, min_height)
    
    local_reachable_pts = np.stack((reachable_r,ca,reachable_z),axis=-1)
    x,y,z = general.calcCartFromCylind(local_reachable_pts)
    local_reachable_pts = np.stack((x,y,z),axis=-1)
    
    return local_reachable_pts
