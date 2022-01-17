#!/usr/bin/env python3
"""
***DESCRIPTION***
"""

"""
IMPORTS
"""
import open3d as o3d
from open3d import *
import numpy as np
import scipy.io as sio
import cv2

"""
METADATA
"""

"""
TODO
"""

"""
CLASS DEFINITIONS
"""

"""
FUNCTIONS DEFINITIONS
"""
def main():

    IR_bin = sio.loadmat('./IrBin.mat')['IR_bin']
    IR_now = sio.loadmat('./IrNow.mat')['IR_now']

    RefImgs = sio.loadmat('./RefImgs.mat')
    IR_ref = RefImgs['IR_ref']
    IR_ind = RefImgs['IR_ind'] - 1 
    
    nLev = 8
    baseRT = 75
    ImgRes = [480, 640]
    corrWind = [9, 9]
    ImgRng = [400, 1000]
    ImgFOV = [45.6, 58.5]

    windSize = int(np.prod(corrWind))
    ImgFOV = np.deg2rad(ImgFOV)

    FocalLength = [ImgRes[1]/(2*np.tan(ImgFOV[1]/2)), ImgRes[0]/(2*np.tan(ImgFOV[0]/2))]

    dOff_min = np.ceil(baseRT * FocalLength[0] / ImgRng[0])
    dOff_max = np.floor(baseRT * FocalLength[0] / ImgRng[1])

    numIntDisp = int(dOff_min - dOff_max + 1)
    disp_all = np.linspace(dOff_min, dOff_max, int((dOff_max-dOff_min)/(-1/nLev))+1)
    depth_all = np.divide(baseRT * FocalLength[0], disp_all)

    DEPTHimg = np.zeros(np.prod(ImgRes))

    for ipix in range(np.prod(ImgRes)):
        
        # Binary window
        window_bin = IR_bin.reshape(-1, order='F')[IR_ind[:,ipix]]

        # Noise window
        window_now = IR_now.reshape(-1, order='F')[IR_ind[:,ipix]]

        if np.sum(window_now) != 0:

            # Estimate integer disparity with binary IR image
            snorm_ref = IR_ref[IR_ind[:,ipix], nLev-1, :]
            snorm_ref = np.reshape(snorm_ref, (windSize, numIntDisp)).astype(bool)
            snorm_now = window_bin - np.sum(window_bin) / windSize
            snorm_now = np.repeat(snorm_now[:, np.newaxis], numIntDisp, axis=1)

            # Maximize horizontal covariance
            horzCov_ref = np.sum(np.multiply(snorm_ref, snorm_now), axis=0)
            dispInd = np.argmax(horzCov_ref)
            dispLookup = (dispInd) * nLev 

            # Sub-pixel refinement with noisy IR image
            window_sub = IR_ref[IR_ind[:,ipix],:,dispInd]
            window_now = np.repeat(window_now[:, np.newaxis], 2*nLev-1, axis=1)

            # Minimize sum of absolute differences
            horzCov_sub = np.sum(np.abs(window_sub - window_now), axis=0)
            dispInd = np.argmin(horzCov_sub)
            dispLookup = dispLookup + dispInd - nLev

            # Convert disparity to depth from lookup table
            DEPTHimg[ipix] = depth_all[dispLookup]

    DEPTHimg = np.reshape(DEPTHimg, ImgRes, order='F')

    # np.savez('cloud.npz', cloud=DEPTHimg)
    
    img = (DEPTHimg / np.max(DEPTHimg) * 255).astype('uint8')
    cv2.imshow('depth', img)
    cv2.waitKey(0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(DEPTHimg.reshape(-1,3))

    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(1)
    o3d.visualization.draw_geometries([pcd , axes])

"""
MAIN
"""
if __name__ == '__main__':
    # main()

    c = np.load('cloud.npz')['cloud']


    x = np.ascontiguousarray(c, dtype=np.float32)    
    depth = o3d.geometry.Image(x)




    intrinsic = o3d.camera.PinholeCameraIntrinsic() # TODO set them
    intrinsic.set_intrinsics(640, 480, 570, 570, 1, 1)
    pcd = o3d.geometry.PointCloud.create_from_depth_image(depth, intrinsic, depth_scale=1)

    # pcd = geometry.create_point_cloud_from_depth_image(depth, intrinsic)
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(1)
    o3d.visualization.draw_geometries([pcd , axes])
    
