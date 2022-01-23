#!/usr/bin/env python3
"""
***DESCRIPTION***
"""

"""
IMPORTS
"""
import os
import argparse
import open3d as o3d
import numpy as np
import scipy.io as sio
import cv2
import freenect

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

    @staticmethod
    def _scaleArray2uint8(array):
        return ((array - array.min()) * (1/(array.max() - array.min()) * 255)).astype('uint8')

    @staticmethod
    def showArray(array, win_name='', delay=0):
        img = KinectSimulator._scaleArray2uint8(array)
        cv2.imshow(win_name, img)
        cv2.waitKey(delay)

    def loadMatFiles(self, folder='./'):

        self.IR_bin = sio.loadmat(os.path.join(folder,'./IrBin.mat'))['IR_bin']
        logger.debug(f'Loaded IR_bin from {folder}')

        self.IR_now = sio.loadmat(os.path.join(folder, './IrNow.mat'))['IR_now']
        logger.debug(f'Loaded IR_now from {folder}')

        RefImgs = sio.loadmat('./RefImgs.mat')
        self.IR_ref = RefImgs['IR_ref']
        logger.debug(f'Loaded IR_ref')

        self.IR_ind = RefImgs['IR_ind'] - 1
        logger.debug(f'Loaded IR_ind')

    def loadNpFiles(self, file):

        RefImgs = np.load('RefImgs.npz')
        self.IR_ref = RefImgs['IR_ref']
        logger.debug(f'Loaded references')

        self.IR_ind = RefImgs['IR_ind'] - 1
        logger.debug(f'Loaded indices')

        object = np.load(file)
        self.IR_bin = object['IR_bin']
        self.IR_now = object['IR_now']
        logger.info(f'Loaded {file}')

    def getIr(self):
        array, timestamp = freenect.sync_get_video(0, freenect.VIDEO_IR_10BIT)

        self.IR_now = np.pad(array, pad_width=4, mode='constant', constant_values=0)
        self.IR_bin = (self.IR_now > 50) * 1024

        logger.info('Gathered IR frame')
        freenect.sync_stop()

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

    def displayPointCloud(self, point_cloud, axis_size=500):
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(axis_size)
        o3d.visualization.draw_geometries([point_cloud , axes],
                                    window_name='Point Cloud',
                                    front=[-0.24, -0.27, -0.9],
                                    lookat=[-25, -3.57, 2127.75],
                                    up=[0.23, -0.95, 0.22],
                                    zoom=0.7)


"""
FUNCTIONS DEFINITIONS
"""
def configLogger():

    logger.propagate = False

    formatter = logging.Formatter('%(asctime)s [%(module)s | %(levelname)s]: %(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(level=logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # fh = logging.FileHandler('log.log')
    # fh.setLevel(level=logging.INFO)
    # fh.setFormatter(formatter)
    # logger.addHandler(fh)

def prettyDepth(depth):
    np.clip(depth, 0, 2**10 - 1, depth)
    depth >>= 2
    depth = depth.astype(np.uint8)
    return depth

"""
MAIN
"""
if __name__ == '__main__':

    configLogger()

    parser = argparse.ArgumentParser(description='Build point clouds simulating the Kinect procedure.')
    parser.add_argument('-s', '--sim', action='store_true', help='Use simulator.')
    parser.add_argument('-f', '--file', type=str, help='File (.npz) containing the object data.')
    args = parser.parse_args()

    if args.sim:
        kinect = KinectSimulator()

        kinect.loadNpFiles(args.file)
        depth_map = kinect.computeDepthMap()

        kinect.showArray(depth_map, win_name='Depth Map', delay=100)
        kinect.showArray(kinect.IR_now, win_name='Speakle Now', delay=100)
        kinect.showArray(kinect.IR_bin, win_name='Speakle Bin', delay=100)

        pcd = kinect.depthMap2PointCloud(depth_map)
        kinect.displayPointCloud(pcd)
    
    else:
        cv2.namedWindow('Depth')
        cv2.namedWindow('Video')
        cv2.namedWindow('IR')

        logger.info('Press ESC in window to stop')

        while True:
            depth = freenect.sync_get_depth()[0]
            cv2.imshow('Depth', prettyDepth(depth))

            video = freenect.sync_get_video()[0]
            cv2.imshow('Video', video[:, :, ::-1])

            ir = freenect.sync_get_video(0, freenect.VIDEO_IR_10BIT)[0]
            cv2.imshow('IR', prettyDepth(ir))

            if cv2.waitKey(10) == 27:
                break

