import numpy as np
import decimal

#############################################
#         General Helper Functions          #
#############################################

# printing scale setup (using metric system)
UNIT_m = 1
UNIT_cm = 100
UNIT_mm = 1000
UNIT_um = 1e6
UNIT_nm = 1e9

# Euler Rotation matrix functions
Rx = lambda a: np.asarray([[1, 0, 0],
                           [0, np.cos(a), -np.sin(a)],
                           [0, np.sin(a), np.cos(a)]])
Ry = lambda a: np.asarray([[np.cos(a), 0, np.sin(a)],
                           [0, 1, 0],
                           [-np.sin(a), 0, np.cos(a)]])
Rz = lambda a: np.asarray([[np.cos(a), -np.sin(a), 0],
                           [np.sin(a), np.cos(a), 0],
                           [0, 0, 1]])


def calcCylindCord(pts):
    """
    Helper function that converts points from carteasian
    to cylindrical coordinates
    """
    r = np.linalg.norm(pts[...,:2], axis=-1)
    theta = np.arctan2(pts[...,1], pts[...,0])%(2*np.pi)
    z = pts[...,2]
    
    return r, theta, z

def calcCartFromCylind(pts):
    """
    Helper function that converts points from cylindrical
    to carteasian coordinates
    """
    x = pts[...,0]*np.cos(pts[...,1])
    y = pts[...,0]*np.sin(pts[...,1])
    z = pts[...,2]
    
    return x, y, z

def calcSphereCord(pts):
    """
    Helper function that converts carteasian points
    back into polar coordinates
    """
    r = np.linalg.norm(pts, axis=-1)
    theta = np.arccos(pts[...,2]/r)%(np.pi)
    phi = np.arctan2(pts[...,1], pts[...,0])%(2*np.pi)
    
    return r, theta, phi

def totuple(a):
    """
    Helper function that converts nested numpy vectors
    into tuples to allow json style saving
    """
    if a.shape == ():
        return a.item()
    else:
        return tuple(map(totuple,a))

def generate_tf(normal):
    """
    Helper function that when given a normal vector
    converts it into a coordinate frame with the normal
    vector as the z-axis. (note that the coordinate frame
    can rotate about the z-axis. additional computation 
    is needed to fix x/y-axis along a direction.
    """
    e3 = normal/np.linalg.norm(normal, keepdims=True)
    e1 = np.zeros(3)
    e1[np.argmin(e3)] = e3[np.argmax(e3)]
    e1[np.argmax(e3)] = -1*e3[np.argmin(e3)]
    e1 = e1/np.linalg.norm(e1, keepdims=True)
    e2 = np.cross(e3,e1)
    e2 = e2/np.linalg.norm(e2, keepdims=True)
    
    return e1, e2, e3



        


