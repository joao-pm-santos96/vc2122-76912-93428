#!/usr/bin/env python3
"""
***DESCRIPTION***
"""

"""
IMPORTS
"""
import scipy.io as sio
import numpy as np
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
class Kinect:

    def __init__(self, 
        nlev = 8,
        corrWind = [9,9],
        isLoadPattern = True,
        isQuant10 = True,
        isQuantOK = True, 
        adjRowShift = 0.5, 
        adjColShift = 0.5,
        ImgRes = [480, 640],
        ImgFOV = [45.6, 58.5],
        ImgRng = [800, 4000],
        baseRT = 75,
        model='default'):
        
        # IR simulator parameters
        self.nlev = nlev # Number of levels to perform interpolation for sub-pixel accuracy
        self.corrWind = corrWind # Size of correlation window used for depth estimation
        self.isLoadPattern = isLoadPattern # Option to load idealized binary replication of the Kinect dot pattern
        self.isQuant10 = isQuant10 # Option to quantize the IR image to 10-bit value
        self.isQuantOK = isQuantOK # If IR intensity model is set to 'none', turn off IR image quantizing
        self.adjRowShift = adjRowShift # Force horizontal lines to be epipolar rectified (in pix)
        self.adjColShift = adjColShift # Force horizontal lines to be epipolar rectified (in pix)

        # Kinect parameters
        self.ImgRes = ImgRes # Resolution of real outputted Kinect IR image (rows x cols)
        self.ImgFOV = ImgFOV # Field of view of transmitter/receiver (vertFOV x horzFOV) (deg)
        self.ImgRng = ImgRng # Minimum and maximum operational depths of the Kinect sensor (min x max) (mm)
        self.baseRT = baseRT # Distance between IR transmitter and receiver (mm)

        # Set input parameters
        self.model_Intensity = None
        if model == 'default':
            self.model_Intensity = lambda i,r,n,l : np.multiply(i*5.90e+08, np.divide(np.sum((-n).conj()*l, axis=0).T, np.power(r,2)))
        elif model == 'simple':
            self.model_Intensity = lambda i,r,n,l : np.divide(i*5.96e+08,np.power(r,2))
        elif model == 'none':
            self.model_Intensity = lambda i,r,n,l : np.multiply(i, np.ones(r.shape))
        else:
            self.model_Intensity = model

        # Preprocess parameters
        self.dotPattern = None
        if self.isLoadPattern:
            self.dotPattern = sio.loadmat('./kinect_pattern_3x3.mat')['dotPattern']
        else:
            raise Exception('Did not load pattern.')

        # IR dot pattern and padded image sizes
        self.ImgResPad = np.add(self.ImgRes,np.subtract(self.corrWind,1))
        self.ImgSizeDot = self.dotPattern.shape

        # Number of pixels in correlation window
        self.windSize = self.corrWind[0]*self.corrWind[1]

    def Preprocess_RefImgs(self):

        # Preprocess indices for reference and noisy IR images
        IR_ind = np.zeros((self.windSize, np.prod(self.ImgRes)))

        # TODO improve performance
        ipix = 0
        for ipix_col in range(self.ImgRes[1]):
            for ipix_row in range(self.ImgRes[0]):

                # Determine indices for correlation window
                row_aux = np.arange(ipix_row, ipix_row + self.corrWind[0])
                row_now = np.tile(row_aux, (self.corrWind[0], 1)).T

                col_aux = np.arange(ipix_col, ipix_col + self.corrWind[1])
                col_now = np.tile(col_aux, (self.corrWind[1],1))

                row_now = row_now.reshape((-1,1), order='A')
                col_now = col_now.reshape((-1,1), order='F')

                ind_now = np.add(row_now, np.multiply(np.subtract(col_now,1), self.ImgResPad[0]))

                # Store values
                IR_ind[:,ipix] = ind_now[:,0]
                ipix += 1

        IR_ind = np.uint32(IR_ind.clip(min=0))

        # Preprocess reference IR images
        # Determine horizontal and vertical focal lengths
        ImgFOV = np.deg2rad(self.ImgFOV)
        FocalLength = [self.ImgRes[1]/(2*np.tan(ImgFOV[1]/2)), self.ImgRes[0]/(2*np.tan(ImgFOV[0]/2))] # pix

        # Number of rows and columns to pad IR image for cross correlation
        corrRow = (self.corrWind[0]-1)/2
        corrCol = (self.corrWind[1]-1)/2

        # Set new depth and find offset disparity for minimum reference image
        dOff_min   = np.ceil(self.baseRT * FocalLength[0] / self.ImgRng[0])
        minRefDpth = self.baseRT * FocalLength[0] / dOff_min

        # Set new depth and find offset disparity for maximum reference image
        dOff_max   = np.floor(self.baseRT * FocalLength[0] / self.ImgRng[1])
        maxRefDpth = self.baseRT * FocalLength[0] / dOff_max

        # Number of disparity levels between min and max depth
        numIntDisp = int(dOff_min - dOff_max + 1)

        # Preprocess depths for all simulated disparities
        # disp_all  = dOff_min:-1/nlev:dOff_max
        disp_all = np.linspace(dOff_min, dOff_max, int((dOff_max-dOff_min)/(-1/self.nlev))+1)
        depth_all = np.divide(self.baseRT * FocalLength[0], disp_all)

        # Add columns of dot pattern to left and right side based on disparity equation
        minDisparity = np.ceil((self.baseRT*FocalLength[0])/minRefDpth)
        maxDisparity = np.floor((self.baseRT*FocalLength[0])/maxRefDpth)

        # Number of cols cannot exceed size of dot pattern (for simplicity of coding)
        pixShftLeft_T = np.min([self.ImgSizeDot[1], np.max([0, np.floor((self.ImgRes[1]-self.ImgSizeDot[1])/2)+1+minDisparity+corrCol])])
        pixShftRght_T = np.min([self.ImgSizeDot[1], np.max([0, np.floor((self.ImgRes[1]-self.ImgSizeDot[1])/2)+1-maxDisparity+corrCol])])
        pixShftLeft_T = int(pixShftLeft_T)
        pixShftRght_T = int(pixShftRght_T)

        # Preprocess parameters for transmitter rays
        # Generage reference image of entire IR pattern projection
        dotAddLeft = self.dotPattern[:,-pixShftLeft_T:]
        dotAddRght = self.dotPattern[:, range(0,pixShftRght_T)]
        dotAdd = np.concatenate((dotAddLeft, self.dotPattern, dotAddRght), axis=1)

        dotIndx = np.argwhere(dotAdd.T==1)
        ImgSizeAdd = dotAdd.shape

        # Convert index to subscript values 
        jpix_x = dotIndx[:,0]
        jpix_y = dotIndx[:,1]

        # Determine where IR dots split to the left of the main pixel
        indxLeft = jpix_x>0
        aux1 = np.subtract(jpix_x[indxLeft], 1)
        aux2 = np.multiply(aux1, ImgSizeAdd[0])
        dotIndxLeft = np.add(jpix_y[indxLeft], aux2)

        indxRght = jpix_x<(ImgSizeAdd[1]-1)
        aux3 = np.multiply(np.add(jpix_x[indxRght],1),ImgSizeAdd[0])
        dotIndxRght = np.add(jpix_y[indxRght], aux3)

        # Crop reference image to fit padded size 
        minFiltRow = 1-corrRow
        maxFiltRow = self.ImgResPad[0]-corrRow
        cropRow = np.max([0, ((self.ImgSizeDot[0] - self.ImgRes[0])/2) + self.adjRowShift-1])

        rowRange = np.add(np.arange(minFiltRow, maxFiltRow+1), cropRow).astype(int)
        colRange = np.arange(1,1 + self.ImgResPad[1]).astype(int)

        # Create angles of rays for each sub-pixel from transmitter
        vertPixLeft_T = np.subtract(np.divide(self.ImgSizeDot[0],2),0.5)
        horzPixLeft_T = np.subtract(np.divide(self.ImgSizeDot[1],2),(0.5-pixShftLeft_T))

        # Adjust subscript value to sensor coordinate system values
        spix_x = np.subtract(horzPixLeft_T, np.add(jpix_x, self.adjColShift))
        spix_y = np.subtract(vertPixLeft_T, np.add(jpix_y, self.adjRowShift))
        
        # Determine 3D location of where each ray intersects unit range
        X_T = np.divide(spix_x, FocalLength[0])
        Y_T = np.divide(spix_y, FocalLength[1])
        Z_T = np.ones(X_T.shape)

        coords = np.column_stack((X_T, Y_T, Z_T))
        XYZ_T = np.linalg.norm(coords, axis=1)

        # Find surface normal for all intersecting dots
        sn_T = np.vstack((np.zeros(X_T.shape),
                                np.zeros(X_T.shape),
                                -1 * np.ones(X_T.shape)))

        # Find the IR light direction for all sub-rays
        ld_T = np.divide(coords.T, XYZ_T)

        # Preprocess fractional intensity arrays
        aux = np.arange(1-1/self.nlev, 1/self.nlev-1/self.nlev, -1/self.nlev)
        leftMain = np.reshape(aux,(1,1,self.nlev-1))
        leftSplt = np.subtract(1, leftMain)

        aux = np.arange(1/self.nlev, (1-1/self.nlev)+1/self.nlev, 1/self.nlev)
        rghtMain = np.reshape(aux,(1,1,self.nlev-1))
        rghtSplt = np.subtract(1, rghtMain)

        # Preprocess reference images with intensities for lookup table
        shape = np.array((np.prod(self.ImgResPad), 2*self.nlev-1, numIntDisp)).astype(int)
        IR_ref = np.zeros(shape)

        for idisp in range(numIntDisp):
            idepth = depth_all[idisp * self.nlev]

            # Compute range of all dots wrt transmitter
            rng_T = idepth*XYZ_T

            # Compute intensities for all dots
            intensity_T = self.model_Intensity(1,rng_T,sn_T,ld_T) # TODO check spix_x and _y

            # Compute reference image where IR dots interesect one pixel
            IR_ref_main = np.zeros(dotAdd.shape)
            IR_ref_main[dotIndx[:,1], dotIndx[:,0]] = intensity_T
            IR_ref_main = (IR_ref_main[:,colRange+idisp])[rowRange,:]

            # Store reference image
            IR_ref[:,self.nlev,idisp] = IR_ref_main.flatten(order='F')

            if idisp == 0:
                # Compute reference images where IR dots split with left pixel
                IR_ref_left = np.zeros(dotAdd.shape)
                np.put(IR_ref_left.T, dotIndxLeft, intensity_T[indxLeft])
                IR_ref_left = (IR_ref_left[rowRange,:])[:,(colRange+idisp)]

                IR_ref_left = np.multiply(np.repeat(IR_ref_main[:,:,np.newaxis], self.nlev-1, axis=2), leftMain) + \
                    np.multiply(np.repeat(IR_ref_left[:,:,np.newaxis], self.nlev-1, axis=2), leftSplt)

                # Store reference images
                IR_ref[:,self.nlev:2*self.nlev-1,idisp] = np.reshape(IR_ref_left, (IR_ref.shape[0], self.nlev-1))

            elif idisp == (numIntDisp - 1):
                # Compute reference images where IR dots split with right pixel
                IR_ref_rght = np.zeros(dotAdd.shape)
                np.put(IR_ref_rght.T, dotIndxRght, intensity_T[indxRght])
                IR_ref_rght = (IR_ref_rght[rowRange,:])[:,(colRange+idisp)]

                IR_ref_rght = np.multiply(np.repeat(IR_ref_main[:,:,np.newaxis], self.nlev-1, axis=2), rghtMain) + \
                    np.multiply(np.repeat(IR_ref_rght[:,:,np.newaxis], self.nlev-1, axis=2), rghtSplt)

                # Store reference images
                IR_ref[:,0:self.nlev-1,idisp] = np.reshape(IR_ref_rght, (IR_ref.shape[0], self.nlev-1))

            else:
                # Compute reference images where IR dots split with left pixel
                IR_ref_left = np.zeros(dotAdd.shape)
                np.put(IR_ref_left.T, dotIndxLeft, intensity_T[indxLeft])
                IR_ref_left = (IR_ref_left[rowRange,:])[:,(colRange+idisp)]

                IR_ref_left = np.multiply(np.repeat(IR_ref_main[:,:,np.newaxis], self.nlev-1, axis=2), leftMain) + \
                    np.multiply(np.repeat(IR_ref_left[:,:,np.newaxis], self.nlev-1, axis=2), leftSplt)

                # Compute reference images where IR dots split with right pixel
                IR_ref_rght = np.zeros(dotAdd.shape)
                np.put(IR_ref_rght.T, dotIndxRght, intensity_T[indxRght])
                IR_ref_rght = (IR_ref_rght[rowRange,:])[:,(colRange+idisp)]

                IR_ref_rght = np.multiply(np.repeat(IR_ref_main[:,:,np.newaxis], self.nlev-1, axis=2), rghtMain) + \
                    np.multiply(np.repeat(IR_ref_rght[:,:,np.newaxis], self.nlev-1, axis=2), rghtSplt)

                # Store reference images
                IR_ref[:,0:self.nlev-1,idisp] = np.reshape(IR_ref_rght, (IR_ref.shape[0], self.nlev-1))
                IR_ref[:,self.nlev:2*self.nlev-1,idisp] = np.reshape(IR_ref_left, (IR_ref.shape[0], self.nlev-1))

        if self.isQuant10 and self.isQuantOK:
            IR_ref = np.round(IR_ref)

        return (IR_ref, IR_ind)


    def KinectSimulator_Depth(self, IR_ref, IR_ind):

        IR_now = cv2.imread('teapod.png', cv2.IMREAD_UNCHANGED)

        IR_bin = np.zeros(IR_now.shape) # TODO...

        DEPTHimg = np.zeros(self.ImgRes)

        for ipix in range(np.prod(self.ImgRes)):

            # Binary window
            # window_bin = IR_bin[IR_ind[:,ipix]]
            window_bin = np.take(IR_bin.T, IR_ind[:,ipix])
            
            # Noisy window
            # window_now = IR_now[IR_ind[:,ipix]]
            window_now = np.take(IR_now.T, IR_ind[:,ipix])

            if np.sum(window_now) != 0:

                # Estimate integer disparity with binary IR image
                snorm_ref = IR_ref[IR_ind[:,ipix],self.nlev,:]
                snorm_ref = np.reshape(snorm_ref, (self.windSize, self.numIntDisp)).astype(bool)
                snorm_now = window_bin - np.sum(window_bin) / self.windSize
                # snorm_now = repmat(snorm_now,1,numIntDisp)



"""
FUNCTIONS DEFINITIONS
"""
def Preprocess_RefImgs(model='default'):

    # IR simulator parameters
    nlev = 8 # Number of levels to perform interpolation for sub-pixel accuracy
    corrWind = [9,9] # Size of correlation window used for depth estimation
    isLoadPattern = True # Option to load idealized binary replication of the Kinect dot pattern
    isQuant10 = True # Option to quantize the IR image to 10-bit value
    isQuantOK = True # If IR intensity model is set to 'none', turn off IR image quantizing
    adjRowShift = 0.5 # Force horizontal lines to be epipolar rectified (in pix)
    adjColShift = 0.5 # Force horizontal lines to be epipolar rectified (in pix)

    # Kinect parameters
    ImgRes = [480, 640] # Resolution of real outputted Kinect IR image (rows x cols)
    ImgFOV = [45.6, 58.5] # Field of view of transmitter/receiver (vertFOV x horzFOV) (deg)
    ImgRng = [800, 4000] # Minimum and maximum operational depths of the Kinect sensor (min x max) (mm)
    baseRT = 75 # Distance between IR transmitter and receiver (mm)

    # Set input parameters
    model_Intensity = None
    if model == 'default':
        model_Intensity = lambda i,r,n,l : np.multiply(i*5.90e+08, np.divide(np.sum((-n).conj()*l, axis=0).T, np.power(r,2)))
    elif model == 'simple':
        model_Intensity = lambda i,r,n,l : np.divide(i*5.96e+08,np.power(r,2))
    elif model == 'none':
        model_Intensity = lambda i,r,n,l : np.multiply(i, np.ones(r.shape))
    else:
        model_Intensity = model   
        
    # Preprocess parameters
    dotPattern = None
    if isLoadPattern:
        dotPattern = sio.loadmat('./kinect_pattern_3x3.mat')['dotPattern']
    else:
        # Force horizontal lines to be epipolar rectified
        raise Exception('Did not load pattern.')

    # IR dot pattern and padded image sizes
    ImgResPad = np.add(ImgRes,np.subtract(corrWind,1))
    ImgSizeDot = dotPattern.shape

    # Number of pixels in correlation window
    windSize = corrWind[0]*corrWind[1]

    # Preprocess indices for reference and noisy IR images
    IR_ind = np.zeros((windSize, np.prod(ImgRes)))

    # TODO improve performance
    ipix = 0
    for ipix_col in range(ImgRes[1]):
        for ipix_row in range(ImgRes[0]):

            # Determine indices for correlation window
            row_aux = np.arange(ipix_row, ipix_row+corrWind[0])
            row_now = np.tile(row_aux, (corrWind[0],1)).T

            col_aux = np.arange(ipix_col, ipix_col+corrWind[1])
            col_now = np.tile(col_aux, (corrWind[1],1))

            row_now = row_now.reshape((-1,1), order='A')
            col_now = col_now.reshape((-1,1), order='F')

            ind_now = np.add(row_now, np.multiply(np.subtract(col_now,1),ImgResPad[0]))

            # Store values
            IR_ind[:,ipix] = ind_now[:,0]
            ipix += 1

    IR_ind = np.uint32(IR_ind.clip(min=0))

    # Preprocess reference IR images
    # Determine horizontal and vertical focal lengths
    ImgFOV = np.deg2rad(ImgFOV)
    FocalLength = [ImgRes[1]/(2*np.tan(ImgFOV[1]/2)), ImgRes[0]/(2*np.tan(ImgFOV[0]/2))] # pix

    # Number of rows and columns to pad IR image for cross correlation
    corrRow = (corrWind[0]-1)/2
    corrCol = (corrWind[1]-1)/2

    # Set new depth and find offset disparity for minimum reference image
    dOff_min   = np.ceil(baseRT*FocalLength[0]/ImgRng[0])
    minRefDpth = baseRT*FocalLength[0]/dOff_min

    # Set new depth and find offset disparity for maximum reference image
    dOff_max   = np.floor(baseRT*FocalLength[0]/ImgRng[1])
    maxRefDpth = baseRT*FocalLength[0]/dOff_max

    # Number of disparity levels between min and max depth
    numIntDisp = int(dOff_min - dOff_max + 1)

    # Preprocess depths for all simulated disparities
    # disp_all  = dOff_min:-1/nlev:dOff_max
    disp_all = np.linspace(dOff_min, dOff_max, int((dOff_max-dOff_min)/(-1/nlev))+1)
    depth_all = np.divide(baseRT*FocalLength[0], disp_all)

    # Add columns of dot pattern to left and right side based on disparity equation
    minDisparity = np.ceil((baseRT*FocalLength[0])/minRefDpth)
    maxDisparity = np.floor((baseRT*FocalLength[0])/maxRefDpth)

    # Number of cols cannot exceed size of dot pattern (for simplicity of coding)
    pixShftLeft_T = np.min([ImgSizeDot[1], np.max([0, np.floor((ImgRes[1]-ImgSizeDot[1])/2)+1+minDisparity+corrCol])])
    pixShftRght_T = np.min([ImgSizeDot[1], np.max([0, np.floor((ImgRes[1]-ImgSizeDot[1])/2)+1-maxDisparity+corrCol])])
    pixShftLeft_T = int(pixShftLeft_T)
    pixShftRght_T = int(pixShftRght_T)

    # Preprocess parameters for transmitter rays
    # Generage reference image of entire IR pattern projection
    dotAddLeft = dotPattern[:,-pixShftLeft_T:]
    dotAddRght = dotPattern[:, range(0,pixShftRght_T)]
    dotAdd = np.concatenate((dotAddLeft, dotPattern, dotAddRght), axis=1)

    # dotIndx = np.argwhere(np.ravel(dotAdd, order='F')==1)
    dotIndx = np.argwhere(dotAdd.T==1)
    ImgSizeAdd = dotAdd.shape

    # Convert index to subscript values 
    # jpix_y = np.fmod(dotIndx, ImgSizeAdd[0])
    # jpix_x = np.divide(np.subtract(dotIndx, jpix_y),ImgSizeAdd[0])
    jpix_x = dotIndx[:,0]
    jpix_y = dotIndx[:,1]

    # Determine where IR dots split to the left of the main pixel
    indxLeft = jpix_x>0
    aux1 = np.subtract(jpix_x[indxLeft], 1)
    aux2 = np.multiply(aux1, ImgSizeAdd[0])
    dotIndxLeft = np.add(jpix_y[indxLeft], aux2)

    indxRght = jpix_x<(ImgSizeAdd[1]-1)
    aux3 = np.multiply(np.add(jpix_x[indxRght],1),ImgSizeAdd[0])
    dotIndxRght = np.add(jpix_y[indxRght], aux3)

    # Crop reference image to fit padded size 
    minFiltRow = 1-corrRow
    maxFiltRow = ImgResPad[0]-corrRow
    cropRow = np.max([0, ((ImgSizeDot[0]-ImgRes[0])/2)+adjRowShift-1])

    rowRange = np.add(np.arange(minFiltRow, maxFiltRow+1), cropRow).astype(int)
    colRange = np.arange(1,1+ImgResPad[1]).astype(int)

    # Create angles of rays for each sub-pixel from transmitter
    vertPixLeft_T = np.subtract(np.divide(ImgSizeDot[0],2),0.5)
    horzPixLeft_T = np.subtract(np.divide(ImgSizeDot[1],2),(0.5-pixShftLeft_T))

    # Adjust subscript value to sensor coordinate system values
    spix_x = np.subtract(horzPixLeft_T, np.add(jpix_x, adjColShift))
    spix_y = np.subtract(vertPixLeft_T, np.add(jpix_y, adjRowShift))
    
    # Determine 3D location of where each ray intersects unit range
    X_T = np.divide(spix_x, FocalLength[0])
    Y_T = np.divide(spix_y, FocalLength[1])
    Z_T = np.ones(X_T.shape)

    coords = np.column_stack((X_T, Y_T, Z_T))
    XYZ_T = np.linalg.norm(coords, axis=1)

    # Find surface normal for all intersecting dots
    sn_T = np.vstack((np.zeros(X_T.shape),
                            np.zeros(X_T.shape),
                            -1 * np.ones(X_T.shape)))

    # Find the IR light direction for all sub-rays
    ld_T = np.divide(coords.T, XYZ_T)

    # Preprocess fractional intensity arrays
    aux = np.arange(1-1/nlev, 1/nlev-1/nlev, -1/nlev)
    leftMain = np.reshape(aux,(1,1,nlev-1))
    leftSplt = np.subtract(1, leftMain)

    aux = np.arange(1/nlev, (1-1/nlev)+1/nlev, 1/nlev)
    rghtMain = np.reshape(aux,(1,1,nlev-1))
    rghtSplt = np.subtract(1, rghtMain)

    # Preprocess reference images with intensities for lookup table
    shape = np.array((np.prod(ImgResPad), 2*nlev-1, numIntDisp)).astype(int)
    IR_ref = np.zeros(shape)

    for idisp in range(numIntDisp):
        idepth = depth_all[idisp*nlev]

        # Compute range of all dots wrt transmitter
        rng_T = idepth*XYZ_T

        # Compute intensities for all dots
        intensity_T = model_Intensity(1,rng_T,sn_T,ld_T) # TODO check spix_x and _y

        # Compute reference image where IR dots interesect one pixel
        IR_ref_main = np.zeros(dotAdd.shape)
        IR_ref_main[dotIndx[:,1], dotIndx[:,0]] = intensity_T
        IR_ref_main = (IR_ref_main[:,colRange+idisp])[rowRange,:]

        # Store reference image
        IR_ref[:,nlev,idisp] = IR_ref_main.flatten(order='F')

        if idisp == 0:
            # Compute reference images where IR dots split with left pixel
            IR_ref_left = np.zeros(dotAdd.shape)
            np.put(IR_ref_left.T, dotIndxLeft, intensity_T[indxLeft])
            IR_ref_left = (IR_ref_left[rowRange,:])[:,(colRange+idisp)]

            IR_ref_left = np.multiply(np.repeat(IR_ref_main[:,:,np.newaxis], nlev-1, axis=2), leftMain) + \
                np.multiply(np.repeat(IR_ref_left[:,:,np.newaxis], nlev-1, axis=2), leftSplt)

            # Store reference images
            IR_ref[:,nlev:2*nlev-1,idisp] = np.reshape(IR_ref_left, (IR_ref.shape[0], nlev-1))

        elif idisp == (numIntDisp - 1):
            # Compute reference images where IR dots split with right pixel
            IR_ref_rght = np.zeros(dotAdd.shape)
            np.put(IR_ref_rght.T, dotIndxRght, intensity_T[indxRght])
            IR_ref_rght = (IR_ref_rght[rowRange,:])[:,(colRange+idisp)]

            IR_ref_rght = np.multiply(np.repeat(IR_ref_main[:,:,np.newaxis], nlev-1, axis=2), rghtMain) + \
                np.multiply(np.repeat(IR_ref_rght[:,:,np.newaxis], nlev-1, axis=2), rghtSplt)

            # Store reference images
            IR_ref[:,0:nlev-1,idisp] = np.reshape(IR_ref_rght, (IR_ref.shape[0], nlev-1))

        else:
            # Compute reference images where IR dots split with left pixel
            IR_ref_left = np.zeros(dotAdd.shape)
            np.put(IR_ref_left.T, dotIndxLeft, intensity_T[indxLeft])
            IR_ref_left = (IR_ref_left[rowRange,:])[:,(colRange+idisp)]

            IR_ref_left = np.multiply(np.repeat(IR_ref_main[:,:,np.newaxis], nlev-1, axis=2), leftMain) + \
                np.multiply(np.repeat(IR_ref_left[:,:,np.newaxis], nlev-1, axis=2), leftSplt)

            # Compute reference images where IR dots split with right pixel
            IR_ref_rght = np.zeros(dotAdd.shape)
            np.put(IR_ref_rght.T, dotIndxRght, intensity_T[indxRght])
            IR_ref_rght = (IR_ref_rght[rowRange,:])[:,(colRange+idisp)]

            IR_ref_rght = np.multiply(np.repeat(IR_ref_main[:,:,np.newaxis], nlev-1, axis=2), rghtMain) + \
                np.multiply(np.repeat(IR_ref_rght[:,:,np.newaxis], nlev-1, axis=2), rghtSplt)

            # Store reference images
            IR_ref[:,0:nlev-1,idisp] = np.reshape(IR_ref_rght, (IR_ref.shape[0], nlev-1))
            IR_ref[:,nlev:2*nlev-1,idisp] = np.reshape(IR_ref_left, (IR_ref.shape[0], nlev-1))

    if isQuant10 and isQuantOK:
        IR_ref = np.round(IR_ref)

    return (IR_ref, IR_ind)


def KinectSimulator_Depth(IR_ref, IR_ind):
    ImgRes = [480, 640] # TODO is duplicated in both functions
    nlev = 8
    corrWind = [9,9]
    windSize = corrWind[0]*corrWind[1]

    ImgRng = [800, 4000]
    ImgFOV = [45.6, 58.5]
    ImgFOV = np.deg2rad(ImgFOV)
    FocalLength = [ImgRes[1]/(2*np.tan(ImgFOV[1]/2)), ImgRes[0]/(2*np.tan(ImgFOV[0]/2))]
    baseRT = 75
    dOff_min   = np.ceil(baseRT*FocalLength[0]/ImgRng[0])
    dOff_max   = np.floor(baseRT*FocalLength[0]/ImgRng[1])
    numIntDisp = int(dOff_min - dOff_max + 1)

    IR_now = cv2.imread('teapod.png', cv2.IMREAD_UNCHANGED)

    IR_bin = np.zeros(IR_now.shape) # TODO...

    DEPTHimg = np.zeros(ImgRes)

    for ipix in range(np.prod(ImgRes)):

        # Binary window
        # window_bin = IR_bin[IR_ind[:,ipix]]
        window_bin = np.take(IR_bin.T, IR_ind[:,ipix])
        
        # Noisy window
        # window_now = IR_now[IR_ind[:,ipix]]
        window_now = np.take(IR_now.T, IR_ind[:,ipix])

        if np.sum(window_now) != 0:

            # Estimate integer disparity with binary IR image
            snorm_ref = IR_ref[IR_ind[:,ipix],nlev,:]
            snorm_ref = np.reshape(snorm_ref, (windSize,numIntDisp)).astype(bool)
            snorm_now = window_bin - np.sum(window_bin) / windSize
            # snorm_now = repmat(snorm_now,1,numIntDisp)
            






"""
MAIN
"""
if __name__ == '__main__':
    # (IR_ref, IR_ind) = Preprocess_RefImgs()
    # KinectSimulator_Depth(IR_ref, IR_ind)

    kin = Kinect()
    (IR_ref, IR_ind) = kin.Preprocess_RefImgs()

    kin.KinectSimulator_Depth(IR_ref, IR_ind)

    
    
    pass