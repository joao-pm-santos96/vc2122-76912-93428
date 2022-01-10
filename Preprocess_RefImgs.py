#!/usr/bin/env python3
"""
***DESCRIPTION***
"""

"""
IMPORTS
"""
import scipy.io as sio
import numpy as np

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
        
    # PREPROCESS PARAMETERS
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

"""
MAIN
"""
if __name__ == '__main__':
    (IR_ref, IR_ind) = Preprocess_RefImgs()
    pass