#!/usr/bin/env python3
"""
***DESCRIPTION***
"""

"""
IMPORTS
"""
import os
import open3d as o3d
import numpy as np
import scipy.io as sio
import cv2

import logging
logger = logging.getLogger(__name__)

"""
METADATA
"""

"""
TODO
"""

"""
CLASS DEFINITIONS
"""
class KinectSimulator:

    def __init__(self,
                nLev = 8,
                baseRT = 75,
                ImgRes = [480, 640],
                corrWind = [9, 9],
                ImgRng = [800, 4000],
                ImgFOV = [45.6, 58.5]):
        
        self.nLev = nLev
        self.baseRT = baseRT
        self.ImgRes = ImgRes
        self.corrWind = corrWind
        self.ImgRng = ImgRng
        self.ImgFOV = ImgFOV

        self.windSize = int(np.prod(self.corrWind))
        self.ImgFOV = np.deg2rad(self.ImgFOV)

        self.FocalLength = [self.ImgRes[1]/(2*np.tan(self.ImgFOV[1]/2)), self.ImgRes[0]/(2*np.tan(self.ImgFOV[0]/2))]

        self.dOff_min = np.ceil(self.baseRT * self.FocalLength[0] / self.ImgRng[0])
        self.dOff_max = np.floor(self.baseRT * self.FocalLength[0] / self.ImgRng[1])

        self.numIntDisp = int(self.dOff_min - self.dOff_max + 1)
        self.disp_all = np.linspace(self.dOff_min, self.dOff_max, int((self.dOff_max - self.dOff_min) / (-1 / self.nLev)) + 1)
        self.depth_all = np.divide(self.baseRT * self.FocalLength[0], self.disp_all)

        logger.info('Kinect Simulator ready')

    def loadFiles(self, folder='./'):

        self.IR_bin = sio.loadmat(os.path.join(folder, './IrBin.mat'))['IR_bin']
        logger.debug(f'Loaded IR_bin from {folder}')

        self.IR_now = sio.loadmat(os.path.join(folder, './IrNow.mat'))['IR_now']
        logger.debug(f'Loaded IR_now from {folder}')

        RefImgs = sio.loadmat(os.path.join(folder, './RefImgs.mat'))
        self.IR_ref = RefImgs['IR_ref']
        logger.debug(f'Loaded IR_ref from {folder}')

        self.IR_ind = RefImgs['IR_ind'] - 1
        logger.debug(f'Loaded IR_ind from {folder}')

    def computeDepthMap(self):

        logger.info('Computing depth map')

        DEPTHimg = np.zeros(np.prod(self.ImgRes))

        for ipix in range(np.prod(self.ImgRes)):
            
            # Binary window
            window_bin = self.IR_bin.reshape(-1, order='F')[self.IR_ind[:,ipix]]

            # Noise window
            window_now = self.IR_now.reshape(-1, order='F')[self.IR_ind[:,ipix]]

            if np.sum(window_now) != 0:

                # Estimate integer disparity with binary IR image
                snorm_ref = self.IR_ref[self.IR_ind[:,ipix], self.nLev-1, :]
                snorm_ref = np.reshape(snorm_ref, (self.windSize, self.numIntDisp)).astype(bool)
                snorm_now = window_bin - np.sum(window_bin) / self.windSize
                snorm_now = np.repeat(snorm_now[:, np.newaxis], self.numIntDisp, axis=1)

                # Maximize horizontal covariance
                horzCov_ref = np.sum(np.multiply(snorm_ref, snorm_now), axis=0)
                dispInd = np.argmax(horzCov_ref)
                dispLookup = (dispInd) * self.nLev 

                # Sub-pixel refinement with noisy IR image
                window_sub = self.IR_ref[self.IR_ind[:,ipix],:,dispInd]
                window_now = np.repeat(window_now[:, np.newaxis], 2 * self.nLev-1, axis=1)

                # Minimize sum of absolute differences
                horzCov_sub = np.sum(np.abs(window_sub - window_now), axis=0)
                dispInd = np.argmin(horzCov_sub)
                dispLookup = dispLookup + dispInd - self.nLev

                # Convert disparity to depth from lookup table
                DEPTHimg[ipix] = self.depth_all[dispLookup]

        DEPTHimg = np.reshape(DEPTHimg, self.ImgRes, order='F')

        return DEPTHimg

    def depthMap2PointCloud(self, depth_map):

        map_array = np.ascontiguousarray(depth_map, dtype=np.float32)    
        depth = o3d.geometry.Image(map_array)

        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        intrinsic.set_intrinsics(self.ImgRes[1], self.ImgRes[0], self.FocalLength[1], self.FocalLength[0], self.ImgRes[1] / 2, self.ImgRes[0] / 2)

        pcd = o3d.geometry.PointCloud.create_from_depth_image(depth, intrinsic, depth_scale=1)

        return pcd

    def saveDepth(self, depth_map, filename='cloud.npz'):
        np.savez(filename, depth_map=depth_map)

    def displayDepthMap(self, depth_map, delay=0):
        # img = (depth_map / np.max(depth_map) * 255).astype('uint8')
        img = ((depth_map - depth_map.min()) * (1/(depth_map.max() - depth_map.min()) * 255)).astype('uint8')
        cv2.imshow('Depth Map', img)

        logger.info('Press any key on image to continue')
        cv2.waitKey(delay)

    def displayPointCloud(self, point_cloud):
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(500)
        o3d.visualization.draw_geometries([point_cloud , axes])


"""
FUNCTIONS DEFINITIONS
"""
def configLogger():

    logger.setLevel(level=logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s [%(module)s | %(levelname)s]: %(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(level=logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # fh = logging.FileHandler('log.log')
    # fh.setLevel(level=logging.INFO)
    # fh.setFormatter(formatter)
    # logger.addHandler(fh)

def mat2npz():
    pass # TODO

"""
MAIN
"""
if __name__ == '__main__':

    configLogger()
    
    # ola = KinectSimulator(ImgRng = [1400, 2000])
    ola = KinectSimulator()    
    
    ola.loadFiles(folder='../KinectSimulator')
    c = ola.computeDepthMap()

    ola.displayDepthMap(c, delay=1)

    pcd = ola.depthMap2PointCloud(c)    
    ola.displayPointCloud(pcd)
    
    
