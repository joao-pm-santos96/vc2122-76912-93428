#!/usr/bin/env python3
"""
KINECTSIMULATOR_DEPTH Simulate Kinect depth images. 
   KINECTSIMULATOR_DEPTH is a program developed to simulate high fidelity
   Kinect depth images by closely following the Microsoft Kinect's 
   mechanics. For a detailed description of how this simulator was 
   developed, please refer to [1]. If this simulator is used for 
   publication, please cite [1] in your references.

   DEPTHimg = KinectSimulator_Depth(vertex,face,normalf) returns the 
   simulated depth image 'DEPTHimg'. The parameters 'vertex' and 'face' 
   define the CAD model of the 3D scene used to generate an image.

       vertex  - 3xn, n vertices of each 3D coordinate that defines the  
                 CAD model.
       face    - 3xm, m facets, each represented by 3 vertices that  
                 defines the CAD model.
       normalf - 3xm, m facets, representing the normal direction of each
                 facet.

   The depth image simulator calls the function KINECTSIMULATOR_IR to
   generate a series of non-noisy reference IR images and the output noisy 
   IR image of the 3D scene to estimate the noisy depth image. Note, The 
   IR image simulator program utilizes a Matlab wrapper written by Vipin 
   Vijayan [2] of the ray casting program OPCODE. The Original OPCODE
   program was written by Pierre Terdiman [3].
   
   Depth image estimation undergoes two steps. The first step estimates an
   integer disparity value by correlating a pixel window of the binary  
   measured image to a binary reference image of a flat wall at either the   
   min or max operational depth. This step finds a best match between a  
   window centered around a pixel of the projected IR pattern from the 
   measurement and reference images. The epipolar lines of the transmitter 
   and receiver are assumed coincident, which means the pattern can only
   shift by a varying number of column pixels. The program then performs a 
   sub-pixel refinement step to estimate a fractional disparity value. 
   This is done by finding the minimum sum of absolute differences between  
   the noisy measured IR image and a non-noisy reference image of a flat 
   wall at a distance computed by the initial integer disparity 
   estimation. This step estimates where the location where IR dots in the  
   window template split pixels. The depth 'z' is computed from a given  
   disparity 'd' by

       z = b*f / d,

   where 'f' is the horizontal focal length of the Kinect sensor, and 'b' 
   is the baseline distance between the transmitter and receiver. The 
   default baseline distance is fixed to 75 mm, and the focal length is 
   computed from the inputted FOV, where the default is 571.4 pixels.

   [DEPTHimg, IRimg] = KinectSimulator_Depth(vertex,face,normalf) returns  
   the simulated noisy IR image IRimg_disp, generated from the 
   KINECTSIMULATOR_IR function. This image has the same size and 
   resolution of the real output Kinect IR image (480 rows x 640 columns). 
   Note, all intersecting IR dots are displayed, regardless of the 
   operational range of depths.
   
   DEPTHimg = KinectSimulator_Depth(vertex,face,normalf,IR_intensity,IR_speckle,IR_detector) 
   allows the user to specify a different IR intensity and noise model for
   the IR image simulator.

       IR_intensity Options
           'default' - IR intensity model determined empirically from data
                       collected from IR images recorded by the Kinect
                       positioned in front of a flat wall at various
                       depths. The model follows an alpha*dot(-n,l)/r^2 
                       intensity model, where r is the range between the
                       sensor and the surface point, n is the surface
                       normal, and l is the IR lighting direction. Alpha 
                       accounts for a constant illumination, surface  
                       properties, etc. The actual intensity model is
                       
                           I = Iu.*5.90x10^8*dot(-n,l)/r^2

                       Iu is the fractional intensity contributed by a
                       sub-ray of a transmitted dot. r is the distance
                       between the center of the transmitter coordinate 
                       system, and the 3D location of where the sub-ray 
                       intersected the CAD model (range).

           'simple'  - Additional IR intensity model determined 
                       empirically from data collected from Kinect IR  
                       images. 

                           I = Iu.*5.96x10^8/r^2

                       This model is a simplified version of the 'default'
                       model, which excluded the surface normal and IR 
                       lighting direction to fit the model.

           'none'    - No model is used to compute an IR intensity. This 
                       is modeled by 

                           I = Iu

                       This option is used when generating reference 
                       images for depth estimation.

       @(Iu,r,n,l)fn - User defined IR intensity model given the
                       fractional intensity, range of the transmitted
                       sub-ray, surface normal, and lighting direction. 

       IR_speckle Options
           'default' - IR speckle noise model determined empirically from 
                       data collected from IR images recorded by the 
                       Kinect positioned in front of a flat wall at 
                       various depths. The model has a multiplicative term 
                       that fits a gamma distribution with shape value 
                       4.54 and scale value 0.196. This noise is added to 
                       each IR dot, separately.

                           I = I*Gamma
                             = I*gamrnd(4.54,0.196)

           'none'    - No model is used to compute an IR noise. This is
                       modeled by 

                           I = Iu

                       This option is used when generating reference 
                       images for depth estimation. Note, this option must
                       be set if the IR_intensity model is set to 'none'.

           @(I)fn    - User defined IR noise model given the intensity of
                       the sub-ray representing a part of the transmitted
                       IR dot.

       IR_detector Options
           'default' - IR detector noise model determined empirically from 
                       data collected from IR images recorded by the 
                       Kinect positioned in front of a flat wall at 
                       various depths. The model has an additive term 
                       that fits a normal distribution with mean -0.126  
                       and standard deviation 10.4, with units of 10-bit 
                       intensity. This noise is added to each pixel,
                       separately.

                           Z = I + n
                             = I - 0.126+10.4*randn()

           'none'    - No model is used to compute an IR noise. This is
                       modeled by 

                           Z = I 

                       This option is used when generating reference 
                       images for depth estimation. Note, this option must
                       be set if the IR_intensity model is set to 'none'.

           @(I)fn    - User defined IR noise model given the intensity of
                       the sub-ray representing a part of the transmitted
                       IR dot.

   DEPTHimg = KinectSimulator_Depth(vertex,face,normalf,IR_intensity,IR_speckle,IR_detector,wallDist) 
   allows the user the option to add a wall to the CAD model of the
   simulated 3D scene.

       wallDist Options
           []        - No wall is added to the scene. This is the default
                       setting.

           'max'     - A flat wall is added to the 3D scene at the maximum
                       operational depth.

           wallDist  - The user can input an single value in millimeters
                       between the min and max operational depths.

   DEPTHimg = KinectSimulator_Depth(vertex,face,normalf,IR_intensity,IR_speckle,IR_detector,wallDist,options) 
   gives the user the option to change default Kinect input parameters and
   the default IR and depth simulator parameters. The options are listed 
   as follows:

       Depth Simulator Parameters
           'refine'  - The interpolation factor to perform sub-pixel
                       refinement. Since the Kinect performs interpolation
                       by a factor of 1/8th of a pixel, the default
                       for this option is set to 8.

           'quant11' - The option to quantize depth values to real
                       allowable 11-bit depth values outputted by the 
                       Kinect sensor. The 'default' option loads an array
                       of depth values collected from real data outputted 
                       by a Kinect for Windows sensor. The user may also
                       set this option to 'off', or input a new array of
                       quantized depth values, all of which must be
                       greater than zero.

         'displayIR' - The user may set this option to 'on' to display
                       the noisy IR image of the 3D scene. The default is
                       set to 'off'.

       IR Simulator Parameters
           'window'  - Size of correlation window used to process IR image
                       images for depth estimation. The default is set to
                       9x9 rows and columns, i.e. [9 9] pixels. Note, 
                       these values must be greater than zero, and must be 
                       odd to allow the pixel being processed to be at the 
                       center of the window. Also, given the limited size 
                       of the idealized dot pattern used to simulate IR 
                       images, the number of rows in the window cannot 
                       exceed 15 pixels.

           'subray'  - Size of sub-ray grid used to simulate the physical
                       cross-sectional area of a transmitted IR dot. The
                       default is set to 7x17 rows and cols, i.e. [7 17]. 
                       Note, it is preferable to set each value to an odd
                       number as to allow the center of the pixel to be 
                       represented by a sub-ray. Also, since Kinect 
                       performs an interpolation to achieve a sub-pixel 
                       disparity accuracy of 1/8th of a pixel, there 
                       should be at least 8 columns of sub-rays.

           'pattern' - The dot pattern used to simulate the IR image. The
                       default is adapted from the work done by Andreas 
                       Reichinger, in his blog post entitled 'Kinect 
                       Pattern Uncovered' [4]. Note, this idealized binary
                       representation of Kinect's dot pattern does not
                       include the pincushion distortion observed in real
                       Kinect IR images.

           'quant10' - The option to quantize IR intensity into a 10-bit
                       value. The default is set to 'on', but can be 
                       turned to 'off'. 

       Kinect Parameters
           'imgfov'  - The field of view of Kinect's transmitter/receiver.
                       The default is set to 45.6 x 58.5 degrees for the 
                       verticle and horizontal FOVs, respectively, i.e. 
                       [45.6 58.5].

           'imgrng'  - The minimum and maximum operational depths of the
                       Kinect sensor. Dots with depths that fall outside 
                       of this range are filtered out from the IRimg and 
                       IRimg_disp images. This is important for the depth
                       image simulator because reference images generated
                       at the min and max depths with only be able to find
                       matches in the simulated measured IR image between
                       the set operational depth range. The default is set
                       to 800 mm for the minimum depth, and 4000 mm for
                       the maximum depth, i.e. [800 4000].

   Notes about the options:
       By limiting disparity estimation to an 1/8th of a pixel, this
       simulator in essence quantizes depth similar to the way Kinect 
       quantizes depth images into 11-bit values. However, the estimated
       horizontal FOV and simulated disparity values most likely differ 
       from the exact Kinect parameters, and therefore setting 'quant11' 
       to 'off' will result in output depths different from real Kinect 
       depth values.

       Keeping the IR image quantization option 'quant10' to 'on' will
       result in introducing more noise to the output IR values on the
       order of 10*log10(2^10) = 30.1 dB, which impacts depth estimation
       in the depth image simulator. Depending on the inputted IR  
       intensity and noise models, this may introduce erroneous depth 
       error, so the user can choose to set this option to 'off' to avoid
       this.        

       If 'imgrng' is set to a larger range, processing will be slower
       because pixel templates of the measured IR image need to be 
       compared to more columns of pixel templates preprocessed reference  
       image array. Also, if the range is smaller, the error in depth 
       image estimates will be smaller.

 References: 
   [1] M. J. Landau, B. Y. Choo, P. A. Beling, “Simulating Kinect Infrared  
       and Depth Images,” IEEE Transactions on Cybernetics. 2015.

   [2] V. Vijayan, “Ray casting for deformable triangular 3d meshes -
       file exchange - MATLAB central,” Apr. 2013.
       http://www.mathworks.com/matlabcentral/fileexchange/41504-ray-casting-for-deformable-triangular-3d-meshes/content/opcodemesh/matlab/opcodemesh.m

   [3] P. Terdiman, “OPCODE,” Aug. 2002. 
       http://www.codercorner.com/Opcode.htm

   [4] A. Reichinger, “Kinect pattern uncovered | azt.tm’s blog,” Mar. 2011.
       https://azttm.wordpress.com/2011/04/03/kinect-pattern-uncovered/
"""

"""
IMPORTS
"""
import numpy as np
import numpy.matlib as matlib
import scipy.io

"""
METADATA
"""
__copyright__ = 'Copyright December2021'
__credits__ = ['Joao Santos', 'Samuel Silva']
__version__ = '1.0.0'
# __license__ = 'GPL'

"""
TODO
"""

"""
CLASS DEFINITIONS
"""
class KinectDepth:

    def __init__(self):

        # Depth simulator parameters
        self._num_levels = 8 # Number of levels to perform interpolation for sub-pixel accuracy
        self._quant_11 = True # Option to quantize depth image into 11-bit value
        self._quant_load = True # Option to use depth quantization model
        self._plot_ir = False # Option to plot IR image from depth image simulator

        # IR simulator parameters
        self._corr_wind = [9, 9] # Size of correlation window used for depth estimation
        self._wall = False # Option to add a wall at depth
        self._load_pattern = True # Option to load idealized binary replication of the Kinect dot pattern
        self._quant_ok = True # If IR intensity model is set to 'none', turn off IR image quantizing


        # Kinect parameters
        self._resolution = (480, 640) # Resolution of real outputted Kinect IR image (rows x cols) [pix]
        self._fov = (45.6, 58.5) # Field of view of transmitter/receiver (vertFOV x horzFOV) [deg]
        self._range = (800, 4000) # Minimum and maximum operational depths of the Kinect sensor (min x max) [mm]
        self._base_rt = 75 # Distance between IR transmitter and receiver [mm]

        # PlaceHolders
        self._dot_pattern = None
        self._dot_pattern_size = None

        # Preprocess parameters
        self._processParams()

        # Load data
        self._loadKinectData()
    
    def _processParams(self):
        
        # Convert FOV to radians
        self._fov = np.deg2rad(self._fov)

        # Compute focal length
        self._focal_length = np.flip(np.divide(self._resolution, 2*np.tan(self._fov/2)))

        # Number of rows and columns to pad IR image for cross correlation
        self._corr_row, self._corr_col = np.divide(np.subtract(self._corr_wind,1),2)

        # Find min and max depths for ref image so dots intersect one pixel
        if self._base_rt * self._focal_length[0]/self._range[1] < 1:
            raise Exception('Maximum depth is too large to compute good max reference image.')

        # Set new depth and find offset disparity for minimum reference image
        self._disp_offset_min = np.ceil(self._base_rt*self._focal_length[0]/self._range[0])
        self._min_reference_depth = self._base_rt * self._focal_length[0] / self._disp_offset_min

        # Set new depth and find offset disparity for maximun reference image
        self._disp_offset_max = np.floor(self._base_rt*self._focal_length[0]/self._range[1])
        self._max_reference_depth = self._base_rt * self._focal_length[0] / self._disp_offset_max

        # Number of disparity levels between min and max depth 
        self._num_int_disp = self._disp_offset_min - self._disp_offset_max + 1

        # Number of pixels in correlation window
        self._wind_size = self._corr_wind[0] * self._corr_wind[1]

        # Preprocess depths for all simulated disparities
        delta = -1/self._num_levels
        self._disp_all = np.arange(self._disp_offset_min, self._disp_offset_max + delta, delta)
        self._depth_all = np.divide(self._base_rt * self._focal_length[0], self._disp_all)

    def _loadKinectData(self):

        # Load idealized binary replication of the Kinect dot pattern
        if self._load_pattern:
            self._dot_pattern = scipy.io.loadmat('kinect_pattern_3x3.mat')['dotPattern'] # TODO fix this

            # Check if depth range provides enough coverage for reference images
            if self._disp_offset_min > self._dot_pattern.shape[1]/3:
                raise Exception('Depth range too large for default dot pattern in order to achieve no pattern overlap.\nHint: Try a minimum depth of at least 204 mm.')

        else:
            # Check if depth range provides enough coverage for reference images
            if self._disp_offset_min > self._dot_pattern.shape[1]:
                raise Exception('Depth range too large for size of dot pattern in order to achieve no pattern overlap.')

            min_row = self._resolution[0] + self._corr_wind[0] - 1

            if self._dot_pattern.shape[0] < min_row:
                raise Exception (f'Dot pattern must have at least {min_row} rows for a correlation window with {self._corr_wind[0]} rows')

        self._dot_pattern_size = self._dot_pattern.shape

    def PreProcessRefImages(self):
        """
        PREPROCESS_REFIMGS Preprocess reference images and window indices. 
        PREPROCESS_REFIMGS is a function called by KINECTSIMULATOR_DEPTH to
        preprocess reference image and index arrays for faster depth image
        processing.

        [IR_REF, IR_IND] = Preprocess_RefImgs(varargin_ref) returns arrays
        IR_REF and IR_IND corresponding to the input parameters specified in
        the KINECTSIMULATOR_DEPTH function call. 

            IR_REF - Contains all reference images at depths corresponding to
                        integer disparities between the minimum and maximum
                        operation depths. This array also contains all possible
                        sub-pixel shifts at each integer disparity depth. These
                        reference images are the same size as the padded IR image
                        returned by the KINECTSIMULATOR_IR function.
                        This array is therefore 3 dimensional, with a size of

                        numel(IRimg) x 2*nlev-1 x numIntDisp

                        where numel(IRimg) is the total number of pixels in the
                        padded IR image, nlev is the number of levels set for 
                        sub-pixel refinement, and numIntDisp is the total number 
                        of integer disparities between the minimum and maximum 
                        operational depths.

            IR_IND - Contains the indices of all pixels within the correlation
                        window centered on each IR image pixel. The IR image in
                        this case is the same size and resolution of the real
                        outputted Kinect IR image (480 rows x 640 columns). 
                        This array is therefore 2 dimensional, with a size of 

                        windSize x numel(IRimg_disp)

                        where windSize is the total number of pixels in the
                        correlation window (i.e. numel(corrWind), and 
                        numel(IRimg_disp) is the total number of pixels in the
                        output Kinect IR image (i.e. 307200 pixels).
        """

        # Load idealized binary replication of the Kinect dot pattern
        if self._dot_pattern is None:
            # Force horizontal lines to be epipolar rectified
            if np.mod(self._dot_pattern_size[0],2) == 0:
                adj_row_shift = 0
            if np.mod(self._dot_pattern_size[1],2) == 0:
                adj_col_shift = 0

        # IR dot pattern and padded image sizes
        resolution_padded = np.subtract(np.add(self._resolution,self._corr_wind), 1)

        # Preprocess indices for reference and noisy IR images
        ir_idx = np.zeros((self._wind_size, np.prod(self._resolution)))

        # TODO improve performance
        ipix = 0
        for ipix_col in range(self._resolution[1]):
            for ipix_row in range(self._resolution[0]):
                
                # Determine indices for correlation window    
                row_aux = np.arange(ipix_row, ipix_row+self._corr_wind[0])
                row_now = np.tile(row_aux, (self._corr_wind[0],1)).T

                col_aux = np.arange(ipix_col, ipix_col+self._corr_wind[1])
                col_now = np.tile(col_aux, (self._corr_wind[1],1))

                row_now = row_now.reshape((-1,1), order='A')
                col_now = col_now.reshape((-1,1), order='F')

                
                ind_now = np.add(row_now, np.multiply(np.subtract(col_now,1),resolution_padded[0]))

                # Store values
                ir_idx[:,ipix] = ind_now[:,0]
                ipix += 1
        
        ir_idx = np.uint32(ir_idx.clip(min=0))

        # Add columns of dot pattern to left and right side based on disparity equation
        min_disparity = np.ceil((self._base_rt*self._focal_length[0])/self._min_reference_depth)
        # max_disparity = np.floor((self._base_rt*self._focal_length[0])/self._max_reference_depth)

        # Number of cols cannot exceed size of dot pattern (for simplicity of coding)
        # pix_shift_left = np.min([self._dot_pattern_size[1], np.max([0, np.floor((self._resolution(1)-self._dot_pattern_size(1))/2)+1+min_disparity+self._corr_col])])
        # pix_shift_right = np.min([self._dot_pattern_size[1], np.max([0, np.floor((self._resolution(1)-self._dot_pattern_size(1))/2)+1-min_disparity+self._corr_col])])

        # Generage reference image of entire IR pattern projection
        

        


"""
FUNCTIONS DEFINITIONS
"""

"""
MAIN
"""
if __name__ == '__main__':
    #pass

    kn = KinectDepth()
    kn.PreProcessRefImages()
