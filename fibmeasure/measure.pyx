"""This is a Cython implementation of various techniques to find the centre of
circular optical fibre images for astronomical fibre positioning feedback via
machine vision cameras. Specifically, it has been written during the design of
the WEAVE pick-and-place fibre positioner for the William Herschel Telescope.

Cython is used here to provide decent performance under Python, and to allow
easier porting to C or C++ in the future.

As always in image processing, there's a gotcha in coordinate systems. The main
routines work in image pixel coordinates (x,y), where (0,0) is the top-left
corner of the top-left pixel in the image. However, some of the underlying
functions operate in array coordinates (i,j), row number first, where (0,0) is
the top-left _element_ in the array. Therefore there is voodoo involved in
converting between these, involving swapping coordinates, adding 0.5 to some
of them, etc. Do be careful.

J.G. 2015
"""

import os
import sys
import circlemaker
import bisc_maskmaker
from scipy.optimize import curve_fit
from image_registration import chi2_shift

import numpy as np
cimport numpy as np

from libc.math cimport ceil
from libc.stdlib cimport malloc, free


def setup(fibreDia, saveDir="fibmeasure_setup_data/", setup_xcorr=False,
          setup_biscuit=False, biscuit_nth=100):
    """Loads all the necessary stuff into memory for the various measurement
    operations in the FibreImage class, which currently are a function of the
    expected fibre diameter in the image.

    Stuff is loaded from the disk location specified, under appropriately-named
    sub-folders. Failure to find the required stuff on disk will mean the stuff
    gets generated. This can take a while in some cases!

    IMPORTANT: It is assumed that the size of the cross-correlation 'ideal
    fibre image' array is (2*fibreDia + 1), so the cropping routine in the
    FibreImage routines must match.

    This needs to be run before attempting any FibreImage methods that need
    things like pre-generated array masks.

    This function has not been converted to Cython because it's just a setup
    routine that is run once, and hence we don't mind if it's a bit slow.

    Args:
        fibreDia: The expected fibre diameter, in pixels
        saveDir: The parent directory for loading/saving stuff
        setup_xcorr: Flag for setting up cross-correlation dependencies
        setup_biscuit:  Flag for setting up biscuit cutter dependencies
        biscuit_nth: The fraction of a pixel to which one wants to measure with
                     the biscuit cutter method; i.e. 20 is 1/20 px precision

    Returns:
        Sweet Fanny Adams.
    """
    if saveDir[-1] != '/':
        saveDir = saveDir + '/'

    if setup_xcorr == True:

        arraySize = (2 * int(fibreDia)) + 1
        arrayCtr = (arraySize/2.0, arraySize/2.0)
        
        global xcorr_fibreDia
        global xcorr_array_ideal

        xcorr_fibreDia = fibreDia

        saveDir_xcorr = saveDir + "xcorr/"
        path = saveDir_xcorr + ("%ix%i_%.2f.npy" % (arraySize, arraySize,
                                                    fibreDia))
        try:
            xcorr_array_ideal = np.load(path)
        except:
            print "No saved array found for cross-correlation"
            print "Will now generate and save a new array..."

            if not os.path.exists(saveDir_xcorr):
                os.makedirs(saveDir_xcorr)

            xcorr_array_ideal = circlemaker.make_circle(
                (arraySize,arraySize),
                fibreDia/2.0,
                arrayCtr,
                255)  # 8-bit image assumed
            np.save(path, xcorr_array_ideal)
            print "Done"

    if setup_biscuit == True:

        arraySize = (2 * int(fibreDia)) + 1
        arrayCtr = (arraySize/2.0, arraySize/2.0)

        global bisc_fibreDia
        global bisc_nth
        global bisc_fullMask
        global bisc_dMask1_coords
        global bisc_dMask1_values
        global bisc_dMaskSub_coords
        global bisc_dMaskSub_values

        bisc_fibreDia = fibreDia
        bisc_nth = biscuit_nth
        
        saveDir_biscuit = saveDir + "biscuit/"
        path = saveDir_biscuit + ("%ix%i_%.2f_%i" % (arraySize, arraySize,
                                                     fibreDia, bisc_nth))
        try:
            bisc_fullMask        = np.load(path + "_fullMask.npy"       )
            bisc_dMask1_coords   = np.load(path + "_dMask1_coords.npy"  )
            bisc_dMask1_values   = np.load(path + "_dMask1_values.npy"  )
            bisc_dMaskSub_coords = np.load(path + "_dMaskSub_coords.npy")
            bisc_dMaskSub_values = np.load(path + "_dMaskSub_values.npy")
        except:
            print "No saved masks found for biscuit cutter"
            print "Will now generate and save new masks..."

            if not os.path.exists(saveDir_biscuit):
                os.makedirs(saveDir_biscuit)

            fibR = fibreDia * 0.5

            bisc_fullMask = np.asarray(
                bisc_maskmaker.make_mask(fibR,(0,0)),
                dtype='float64')

            bisc_dMask1 = bisc_maskmaker.make_many_diffmasks(fibR, 1)
            bisc_dMask1_coords = np.asarray(
                bisc_dMask1[0],
                dtype='int32')
            bisc_dMask1_values = np.asarray(
                bisc_dMask1[1],
                dtype='float64')

            bisc_dMaskSub = bisc_maskmaker.make_many_diffmasks(fibR, bisc_nth)
            bisc_dMaskSub_coords = np.asarray(
                bisc_dMaskSub[0],
                dtype='int32')
            bisc_dMaskSub_values = np.asarray(
                bisc_dMaskSub[1],
                dtype='float64')
            
            np.save(path + "_fullMask.npy"       , bisc_fullMask       )
            np.save(path + "_dMask1_coords.npy"  , bisc_dMask1_coords  )
            np.save(path + "_dMask1_values.npy"  , bisc_dMask1_values  )
            np.save(path + "_dMaskSub_coords.npy", bisc_dMaskSub_coords)
            np.save(path + "_dMaskSub_values.npy", bisc_dMaskSub_values)
            
            print "Done"


cdef class FibreImage:

    cdef public:
        unsigned char[:,::1] im_raw
        double[:] refPoint
        int px_thresh_override
        unsigned char nBits, maxVal
        unsigned int[:] cropCentre
        unsigned char[:,::1] im_cropped
        unsigned int cropBorder
        unsigned int[:] im_proj_x
        unsigned int[:] im_proj_y
        unsigned int[:] px_hist
        unsigned int px_nZero, px_nSat
        unsigned char px_valMin, px_valMax, px_valLo, px_valHi
        unsigned char px_mode, px_ptile, px_thresh

    def __cinit__(self, unsigned char[:,::1] imar, double[:] refPoint,
                  int px_thresh_override=-1):
        """FibreImage instances are each based on a single input image. The
        instance can then have various routines (inc. different centre-finding
        techniques) performed on it, and can be destroyed when you're done.

        Right now only 8-bit images are expected.

        Args:
            imar: The image from the camera, as a 2D array, in 8-bit mono, with
                  the image x-axis being the array j-axis and y being i
            refPoint: The pixel coordinate used as the origin when returning
                      measurement results, in [x,y] coords; also used when
                      excluding multiple fibres, as the fibre closest to this
                      point will be treated as the main fibre in the image
            px_thresh_override: The pixel value taken as the zero-threshold
                                for centroiding etc. Setting a negative value
                                will mean this is calculated automatically,
                                based on image stats
        """
        if ((refPoint[0] < 0) or (refPoint[0] > imar.shape[1])
                or (refPoint[1] < 0) or (refPoint[1] > imar.shape[0])):
            raise FibreImageError("Centre reference coordinates outside image")

        self.im_raw = imar
        self.refPoint = refPoint
        self.px_thresh_override = px_thresh_override
        self.nBits = 8
        self.maxVal = 2**self.nBits - 1

        if (self.px_thresh_override > self.maxVal):
            raise FibreImageError("Threshold value exceeds pixel value range")

    cpdef get_centre_centroid(self, float fibreDia,
                              unsigned int coarseSample=4,
                              bint speedTest=False, speedTestRuns=10):
        """This method gets the centre coordinates of a fibre image using the
        centroid method.

        There is no need to call anything before this; all steps are included.

        The expected fibre diameter is used for >=3 things: i) it is passed
        to the preprocessing method in order to guess what the black-point
        threshold of the image is; ii) it is used to guess what the sum of
        the pixel moments should be for a single-fibre image; and iii) it is
        used to decide what the size of the cropped image around the initial
        centroid should be, so that the whole fibre comfortably fits in it.

        Args:
            fibreDia: The approximate expected fibre diameter, in pixels
            coarseSample: The sampling interval for the initial coarse passes
                          through the image; e.g. 2 looks at every other pixel
            speedTest: Flag to run the centre-finding code multiple times and
                       return the average time, using the timeit module. Pre-
                       processing is not included in this result, and the
                       result is not returned, simply printed to stdout.
            speedTestRuns: How many times to run the centre-finding function
                           when doing a speed test. Higher is more accurate.

        Returns:
            An array of the coordinates of the calculated centre of the fibre
            image, in image coordinates [x,y], relative to refPoint
        """
        cdef double[:] result_cropped
        cdef unsigned int i

        if speedTest == False:
            try:
                self.preprocess(fibreDia, coarseSample)
                result_cropped = self.centroid_2D(
                    self.im_cropped,
                    self.px_thresh,
                    sampleStep=1,
                    binary=False)
            except:
                print sys.exc_info()[0], sys.exc_info()[1]
                return None
        else:
            import time
            loopList = range(speedTestRuns)
            timeBegin = time.time()
            for i in loopList:
                self.preprocess(fibreDia, coarseSample)
            timeEnd = time.time()
            timeAverage = (timeEnd-timeBegin) / speedTestRuns
            print "Approx. time pre-processing: %.2f ms" % (timeAverage*1000)
            timeBegin = time.time()
            for i in loopList:
                result_cropped = self.centroid_2D(
                self.im_cropped,
                self.px_thresh,
                sampleStep=1,
                binary=False)
            timeEnd = time.time()
            timeAverage = (timeEnd-timeBegin) / speedTestRuns
            print "Approx. time centre-finding: %.2f ms" % (timeAverage*1000)

        # Note the mixing of (x,y) an (i,j) coordinates here:
        cdef double result_x = self.cropCentre[1]-self.cropBorder \
                               + result_cropped[1]+0.5 \
                               - self.refPoint[0]
        cdef double result_y = self.cropCentre[0]-self.cropBorder \
                               + result_cropped[0]+0.5 \
                               - self.refPoint[1]
        cdef double[:] result = np.array(([result_x,result_y]),
                                         dtype='float64')
        return result

    cpdef get_centre_xcorr(self, unsigned int coarseSample=4,
                           bint speedTest=False, speedTestRuns=10):
        """This method gets the centre coordinates of a fibre image using the
        cross-correlation method. You must run the 'setup' function before
        calling this.

        There is no need to call anything before this; all steps are included.

        The expected fibre diameter is taken from the global value assigned
        when running the 'setup' function, in order to avoid conflicts.

        The expected fibre diameter is used for >=3 things: i) it is passed
        to the preprocessing method in order to guess what the black-point
        threshold of the image is; ii) it is used to guess what the sum of
        the pixel moments should be for a single-fibre image; and iii) it is
        used to decide what the size of the cropped image around the initial
        centroid should be, so that the whole fibre comfortably fits in it.

        Args:
            coarseSample: The sampling interval for the initial coarse passes
                          through the image; e.g. 2 looks at every other pixel
            speedTest: Flag to run the centre-finding code multiple times and
                       return the average time, using the timeit module. Pre-
                       processing is not included in this result, and the
                       result is not returned, simply printed to stdout.
            speedTestRuns: How many times to run the centre-finding function
                           when doing a speed test. Higher is more accurate.

        Returns:
            An array of the coordinates of the calculated centre of the fibre
            image, in image coordinates [x,y], relative to refPoint
        """
        # Check for required globals from 'setup' function:
        if not globals().has_key("xcorr_array_ideal"):
            raise FibreImageError("Cross-correlation reference not loaded")

        cdef float fibreDia = <float>xcorr_fibreDia

        cdef double[:] result_cropped
        cdef unsigned int i

        if speedTest == False:
            try:
                self.preprocess(fibreDia, coarseSample)
                result_cropped = self.xcorr(self.im_cropped, fibreDia)
            except:
                print sys.exc_info()[0], sys.exc_info()[1]
                return None
        else:
            import time
            loopList = range(speedTestRuns)
            timeBegin = time.time()
            for i in loopList:
                self.preprocess(fibreDia, coarseSample)
            timeEnd = time.time()
            timeAverage = (timeEnd-timeBegin) / speedTestRuns
            print "Approx. time pre-processing: %.2f ms" % (timeAverage*1000)
            timeBegin = time.time()
            for i in loopList:
                result_cropped = self.xcorr(self.im_cropped, fibreDia)
            timeEnd = time.time()
            timeAverage = (timeEnd-timeBegin) / speedTestRuns
            print "Approx. time centre-finding: %.2f ms" % (timeAverage*1000)

        # Note the mixing of (x,y) an (i,j) coordinates here:
        cdef double result_x = self.cropCentre[1]-self.cropBorder \
                               + result_cropped[0] \
                               - self.refPoint[0]
        cdef double result_y = self.cropCentre[0]-self.cropBorder \
                               + result_cropped[1] \
                               - self.refPoint[1]
        cdef double[:] result = np.array(([result_x,result_y]),
                                         dtype='float64')
        return result

    cpdef get_centre_biscuit(self, unsigned int coarseSample=4,
                            bint speedTest=False, speedTestRuns=10):
        """This method gets the centre coordinates of a fibre image using the
        biscuit cutter method. You must run the 'setup' function before calling
        this.

        There is no need to call anything before this; all steps are included.

        The expected fibre diameter and the requested precision (nth) is taken
        from the global values assigned when running the 'setup' function, in
        order to avoid conflicts.

        The expected fibre diameter is used for >=3 things: i) it is passed
        to the preprocessing method in order to guess what the black-point
        threshold of the image is; ii) it is used to guess what the sum of
        the pixel moments should be for a single-fibre image; and iii) it is
        used to decide what the size of the cropped image around the initial
        centroid should be, so that the whole fibre comfortably fits in it.

        Args:
            coarseSample: The sampling interval for the initial coarse passes
                          through the image; e.g. 2 looks at every other pixel
            speedTest: Flag to run the centre-finding code multiple times and
                       return the average time, using the timeit module. Pre-
                       processing is not included in this result, and the
                       result is not returned, simply printed to stdout.
            speedTestRuns: How many times to run the centre-finding function
                           when doing a speed test. Higher is more accurate.

        Returns:
            An array of the coordinates of the calculated centre of the fibre
            image, in image coordinates [x,y], relative to refPoint
        """
        # Check for required globals from 'setup' function:
        if not (globals().has_key("bisc_fullMask")
                and globals().has_key("bisc_dMask1_coords")
                and globals().has_key("bisc_dMask1_values")
                and globals().has_key("bisc_dMaskSub_coords")
                and globals().has_key("bisc_dMaskSub_values") ):
            raise FibreImageError("Biscuit cutter masks not loaded")

        cdef float fibreDia = <float>bisc_fibreDia
        cdef unsigned int nth = <unsigned int>bisc_nth

        cdef double[:] result_cropped
        cdef unsigned int i

        if speedTest == False:
            try:
                self.preprocess(fibreDia, coarseSample)
                result_cropped = self.biscuit(self.im_cropped, fibreDia, nth)
            except:
                print sys.exc_info()[0], sys.exc_info()[1]
                return None
        else:
            import time
            loopList = range(speedTestRuns)
            timeBegin = time.time()
            for i in loopList:
                self.preprocess(fibreDia, coarseSample)
            timeEnd = time.time()
            timeAverage = (timeEnd-timeBegin) / speedTestRuns
            print "Approx. time pre-processing: %.2f ms" % (timeAverage*1000)
            timeBegin = time.time()
            for i in loopList:
                result_cropped = self.biscuit(self.im_cropped, fibreDia, nth)
            timeEnd = time.time()
            timeAverage = (timeEnd-timeBegin) / speedTestRuns
            print "Approx. time centre-finding: %.2f ms" % (timeAverage*1000)
        # Note the mixing of (x,y) an (i,j) coordinates here:
        cdef double result_x = self.cropCentre[1]-self.cropBorder \
                               + result_cropped[1]+0.5 \
                               - self.refPoint[0]
        cdef double result_y = self.cropCentre[0]-self.cropBorder \
                               + result_cropped[0]+0.5 \
                               - self.refPoint[1]
        cdef double[:] result = np.array(([result_x,result_y]),
                                         dtype='float64')
        return result

    cpdef get_centre_fitparab(self, float fibreDia,
                              unsigned int coarseSample=4,
                              bint speedTest=False, speedTestRuns=10):
        """This method gets the centre coordinates of a fibre image by fitting
        parabola to the collapsed image data in x and y.

        There is no need to call anything before this; all steps are included.

        The expected fibre diameter is used for >=3 things: i) it is passed
        to the preprocessing method in order to guess what the black-point
        threshold of the image is; ii) it is used to guess what the sum of
        the pixel moments should be for a single-fibre image; and iii) it is
        used to decide what the size of the cropped image around the initial
        centroid should be, so that the whole fibre comfortably fits in it.

        Args:
            fibreDia: The approximate expected fibre diameter, in pixels
            coarseSample: The sampling interval for the initial coarse passes
                          through the image; e.g. 2 looks at every other pixel
            speedTest: Flag to run the centre-finding code multiple times and
                       return the average time, using the timeit module. Pre-
                       processing is not included in this result, and the
                       result is not returned, simply printed to stdout.
            speedTestRuns: How many times to run the centre-finding function
                           when doing a speed test. Higher is more accurate.

        Returns:
            An array of the coordinates of the calculated centre of the fibre
            image, in image coordinates [x,y], relative to refPoint
        """
        cdef double[:] result_cropped
        cdef unsigned int i

        if speedTest == False:
            try:
                self.preprocess(fibreDia, coarseSample)
                result_cropped = self.fitparab(self.im_cropped, fibreDia)
            except:
                print sys.exc_info()[0], sys.exc_info()[1]
                return None
        else:
            import time
            loopList = range(speedTestRuns)
            timeBegin = time.time()
            for i in loopList:
                self.preprocess(fibreDia, coarseSample)
            timeEnd = time.time()
            timeAverage = (timeEnd-timeBegin) / speedTestRuns
            print "Approx. time pre-processing: %.2f ms" % (timeAverage*1000)
            timeBegin = time.time()
            for i in loopList:
                result_cropped = self.fitparab(self.im_cropped, fibreDia)
            timeEnd = time.time()
            timeAverage = (timeEnd-timeBegin) / speedTestRuns
            print "Approx. time centre-finding: %.2f ms" % (timeAverage*1000)
        # Note the mixing of (x,y) an (i,j) coordinates here:
        cdef double result_x = self.cropCentre[1]-self.cropBorder \
                               + result_cropped[1]+0.5 \
                               - self.refPoint[0]
        cdef double result_y = self.cropCentre[0]-self.cropBorder \
                               + result_cropped[0]+0.5 \
                               - self.refPoint[1]
        cdef double[:] result = np.array(([result_x,result_y]),
                                         dtype='float64')
        return result

    ####

    cdef preprocess(self, float fibreDia, unsigned int coarseSample):
        """This is the entire 'pre-processing' stage of the centre-finding
        task. This function is responsible for calling a bunch of more generic
        functions and, most importantly, interpreting their result. This
        function detects the approximate position of the fibre in the image,
        or, if multiple or no fibres are suspected, does what's necessary to
        deal with this.  The final result is the creation of the small sub-
        image array, 'self.im_cropped', which is what the more serious centre-
        finding routines will work on.

        Args:
            fibreDia: The approximate expected fibre diameter, in pixels
            coarseSample: The sampling interval for the initial coarse passes
                          through the image; e.g. 2 looks at every other pixel

        Returns:
            Sweet Fanny Adams.
        """
        self.get_stats(sampleStep=coarseSample, fibreDia=fibreDia)

        # Manually override the calculated threshold value if valid:
        if (self.px_thresh_override >= 0):
            self.px_thresh = self.px_thresh_override

        # Check signal wrt background (mode) and reject if <10% full scale:
        cdef unsigned char minSignal = <unsigned char>(self.maxVal * 0.1)
        if (self.px_valHi-self.px_mode) < minSignal:
            raise FibreImageError("Weak signal")  # Possibly no fibres present

        cdef double[:] fibctr_coarse = self.centroid_2D(
            self.im_raw,
            self.px_thresh,
            sampleStep=coarseSample,
            binary=True)
        cdef unsigned int multiFibThreshold = <unsigned int>(fibreDia**2 *
                                                             fibreDia/2)
        cdef unsigned int[:] multiFibMoments = self.sum_abs_moments_2D(
            self.im_raw,
            self.px_thresh,
            sampleStep=coarseSample,
            centroid=fibctr_coarse,
            binary=True)
        self.cropCentre = np.array(([fibctr_coarse[0],fibctr_coarse[1]]),
                                   dtype='uint32')  # Not an elegant conversion
        cdef unsigned char cropCentre_pxVal = self.im_raw[self.cropCentre[0],
                                                          self.cropCentre[1]]
        if ( (np.amax(multiFibMoments) > multiFibThreshold)
                or (cropCentre_pxVal < self.px_thresh) ):
            print "Warning: Multiple fibres detected"
            self.cropCentre = self.multifib_crop(
                self.im_raw,
                np.array(
                    ([self.refPoint[1],self.refPoint[0]]),
                    dtype='uint32'),  # Not an elegant conversion :(
                <unsigned int>fibreDia,
                fibreDia,
                coarseSample)
        self.cropBorder = <int>fibreDia
        self.im_cropped = self.crop(self.im_raw, self.cropCentre,
                                    self.cropBorder)

    cdef get_stats(self, unsigned int sampleStep, float fibreDia):
        """Various useful things are computed, using a coarse pass through
        the image array: a histogram, stats about the pixel values, also
        a 'collapsed' or 'projected' versions of the image along x and y.

        Args:
            sampleStep: The sampling interval; e.g. 2 for every other element
            fibreDia: The expected diameter of the fibre(s) in the image, in
                      pixels, used to calculate the correct percentile of
                      pixels that 'we care about'

        Returns:
            Sweet Fanny Adams.
        """
        self.px_hist   = np.zeros(self.maxVal+1, dtype='uint32')
        self.px_valMin = 0  # Min pixel value
        self.px_valMax = 0  # Max pixel value
        self.px_nZero  = 0  # Number of pixels with a value of zero
        self.px_nSat   = 0  # Number of saturated pixels
        self.px_valLo  = 0  # Min non-zero value (ex dead pixels)
        self.px_valHi  = 0  # Max non-saturated value (ex hot pixels)
        self.px_mode   = 0  # Most common pixel value
        self.px_ptile  = 0  # The pixel value above which 'we care', based
                               # on the percentile of pixels for the fibre size
        self.px_thresh = 0  # The calculated best pixel value to use as a
                               # threshold for centroid calculations etc.
        self.im_proj_x = np.zeros(int(ceil(self.im_raw.shape[1]/sampleStep)),
                                  dtype='uint32')
        self.im_proj_y = np.zeros(int(ceil(self.im_raw.shape[0]/sampleStep)),
                                  dtype='uint32')

        # Slice through the image, adding the values to the histogram and
        # the projected versions of the image:
        cdef unsigned int x, y, i, j
        cdef unsigned char pxVal
        for x in range(self.im_proj_x.shape[0]):
            j = x * sampleStep
            for y in range(self.im_proj_y.shape[0]):
                i = y * sampleStep
                pxVal = self.im_raw[i][j]
                self.px_hist[pxVal] = self.px_hist[pxVal] + 1
                self.im_proj_x[x] = self.im_proj_x[x] + pxVal
                self.im_proj_y[y] = self.im_proj_y[y] + pxVal
        
        # Ascertain the number of zero-value and saturated pixels, and the
        # otherwise minimum and maximum pixel values:
        self.px_nZero = self.px_hist[0]
        self.px_nSat  = self.px_hist[self.maxVal]

        cdef int k
        for k in range(1,self.maxVal+1,1):
            if self.px_hist[k] > 0:
                self.px_valLo = k
                break
        if self.px_nZero > 0:
            self.px_valMin = 0
        else:
            self.px_valMin = self.px_valLo
        
        for k in range(self.maxVal-1,-1,-1):
            if self.px_hist[k] > 0:
                self.px_valHi = k
                break
        if self.px_nSat > 0:
            self.px_valMax = self.maxVal
        else:
            self.px_valMax = self.px_valHi
        
        # Find the most common pixel value (the mode):
        cdef unsigned int modeCount = 0
        for k in range(self.maxVal+1):
            if self.px_hist[k] > modeCount:
                modeCount = self.px_hist[k]
                self.px_mode = k
        # The appropriate 'percentile' above which we care is taken as approx.
        # one fibre's worth of pixels in the (coarsely-sampled) image:
        cdef unsigned int npx_fibres = <unsigned int>((fibreDia/sampleStep)**2)
        cdef unsigned int totalpx = 0
        for k in range(self.maxVal,-1,-1):
            totalpx += self.px_hist[k]
            if totalpx >= npx_fibres:
                self.px_ptile = k
                break
        # The most suitable pixel value to use as a threshold between dark and
        # light pixels is half way between the mode and the percentile value:
        self.px_thresh = int((self.px_mode + self.px_ptile) / 2)

    cdef crop(self, unsigned char[:,::1] array, unsigned int[:] cropCentre,
              unsigned int cropBorder):
        """Crops an array around a central element, with a border of a
        specified number of elements.

        Args:
            array: A 2D array of values to process
            cropCentre: An array of the array coordinate [i,j] to crop around
            cropBorder: The number of elements each side of the centre (+- i,j)

        Returns:
            A new array of the cropped region
        """
        cdef unsigned int croppedSize = (2*cropBorder) + 1
        cdef unsigned char[:,::1] array_cropped = np.empty(
            shape=(croppedSize,croppedSize), dtype='uint8')

        cdef unsigned int i, j, array_i, array_j
        for i in range(croppedSize):
            for j in range(croppedSize):
                array_i = <unsigned int>cropCentre[0] - cropBorder + i
                array_j = <unsigned int>cropCentre[1] - cropBorder + j
                array_cropped[i,j] = array[array_i,array_j]

        return array_cropped

    cdef multifib_crop(self, unsigned char[:,::1] array,
                       unsigned int[:] cropRef, unsigned int cropChunk,
                       float fibreDia, unsigned int sampleStep):
        """Takes an image array containing multiple fibres and progressively
        crops by a specified amount, about a specified centre, until it appears
        that only one fibre remains. Hence, the remaining fibre will always be
        the one closest to the specified centre point.

        Cropping will be performed asymmetrically until the cropped image is
        square about the specified centre, then will be symmetrical.

        The aim is not to completely remove all trace of other fibres, but to
        confidently arrive at a pixel coordinate that the 'main' fibre is
        under. This coordinate is returned so that the image array can be
        cropped around it before normal centre-finding takes place. This final
        crop should almost definitely remove the other fibres.

        We decide that a pixel coordinate is 'on top of the main fibre' if it
        satisfies two conditions: i) the sum of the absolute moments about the
        image centroid is sufficiently low, and ii) the pixel value at the
        coordinate is above the 'px_thresh' value (i.e. is not just a
        background-level pixel).

        The 'preprocess' method needs to have been run on thr input image
        before calling this, as it relies on the statistical values from it.

        Args:
            array: A 2D array of values to process
            cropRef: Coordinates of a point to crop around ([i,j])
            cropChunk: How many pixels to remove from each side of the image
                       in each iteration of the progressive cropping
            fibreDia: The expected diameter of the fibre(s) in the image, in
                      pixels, used to estimate the threshold acceptable value
                      for the sum of absolute moments calculation
            sampleStep: The sampling interval for centroiding etc.

        Returns:
            An array of a pair of coordinates for a pixel which is thought to
            be directly above the fibre in the image that is closest
            to the image instance's specified refPoint:
            [0]: coordinate along i-axis
            [1]: coordinate along j-axis
        """
        cdef double[:] arrayCentroid = self.centroid_2D(
            array,
            self.px_thresh,
            sampleStep=sampleStep,
            binary=True)
        cdef unsigned int multiFibThreshold = <unsigned int>(fibreDia**2 *
                                                             fibreDia/2)
        cdef unsigned int[:] multiFibMoments = self.sum_abs_moments_2D(
            array,
            self.px_thresh,
            sampleStep=sampleStep,
            centroid=arrayCentroid,
            binary=True)
        cdef unsigned int[:] pxCoords = np.array(
            ([arrayCentroid[0],arrayCentroid[1]]),
            dtype='uint32')  # Not an elegant conversion :(

        cdef unsigned char pxCoords_pxVal = array[pxCoords[0],pxCoords[1]]

        # The initial crop 'margin' is an imaginary square drawn around cropRef
        cdef unsigned int cropMargin = max(cropRef[0],
                                           cropRef[1],
                                           array.shape[0]-1 - cropRef[0],
                                           array.shape[1]-1 - cropRef[1])
        cdef int cropTL_i
        cdef int cropTL_j
        cdef int cropBR_i
        cdef int cropBR_j
        cdef unsigned char[:,::1] array_cropped

        while ((max(multiFibMoments[0],multiFibMoments[1]) > multiFibThreshold)
               or (pxCoords_pxVal < self.px_thresh)):

            cropMargin -= cropChunk

            if cropMargin < 0:
                raise FibreImageError("Could not locate main fibre")

            cropTL_i = max(<int>cropRef[0] - <int>cropMargin, 0)
            cropTL_j = max(<int>cropRef[1] - <int>cropMargin, 0)
            cropBR_i = min(<int>cropRef[0] + <int>cropMargin, array.shape[0]-1)
            cropBR_j = min(<int>cropRef[1] + <int>cropMargin, array.shape[1]-1)

            array_cropped = array[cropTL_i:cropBR_i+1, cropTL_j:cropBR_j+1]

            # Recalculate the fibre moments etc. for the newly cropped array:
            arrayCentroid = self.centroid_2D(
                array_cropped,
                self.px_thresh,
                sampleStep=sampleStep,
                binary=True)
            multiFibMoments = self.sum_abs_moments_2D(
                array_cropped,
                self.px_thresh,
                sampleStep=sampleStep,
                centroid=arrayCentroid,
                binary=True)
            pxCoords = np.array(
                ([arrayCentroid[0],arrayCentroid[1]]),
                dtype='uint32')  # Not an elegant conversion :(
            pxCoords_pxVal = array_cropped[pxCoords[0],pxCoords[1]]

        cdef unsigned int result_i = pxCoords[0] + <unsigned int>abs(cropTL_i)
        cdef unsigned int result_j = pxCoords[1] + <unsigned int>abs(cropTL_j)
        cdef unsigned int[:] result = np.array(
            ([result_i,result_j]),
            dtype='uint32')
        return result

    cdef centroid_1D(self, unsigned char[:] array, unsigned char threshold):
        """Calculate the centroid of a 1D array of values.

        Note that the result is returned in terms of array coordinates. If the
        array happens to represent the pixels of an image, then 0.5 will need
        to be added to the result to convert to image pixel coordinates.

        Args:
            array: A 1D array of values for which to find the centroid
            threshold: The pixel value above which a pixel will be considered

        Returns:
            The position of the centroid in the array
        """
        cdef unsigned int M0 = 0  # Zero'th order moment
        cdef unsigned int M1 = 0  # First order moment

        cdef unsigned int i
        for i in range(array.shape[0]):
            if i > threshold:
                M0 += array[i]
                M1 += array[i] * i

        cdef double result = <double>(1.0 * M1/M0)
        return result

    cdef centroid_2D(self, unsigned char[:,::1] array, unsigned char threshold,
                     unsigned int sampleStep, bint binary=True):
        """Calculates the centroid of a 2D array.

        The array is handled by either weighting by the values in the array
        (that are above the threshold), or by giving all elements above the
        threshold equal weight (i.e. binary).

        Note that the result is returned in terms of array coordinates. If the
        array happens to represent an image, then 0.5 will need to be added to
        each of the i and j results to convert to image pixel coordinates.

        Args:
            array: A 2D array of values to analyse
            threshold: The value above which elements will be considered
            sampleStep: The sampling interval; e.g. 2 for every other element
            binary: Flag to give all values above threshold equal weight

        Returns:
            An array of the centroid location, in array coordinates:
            [0]: along i-axis
            [1]: along j-axis
        """
        cdef unsigned int M00 = 0  # Zero'th order moment
        cdef unsigned int M01 = 0  # First order moment for i-axis
        cdef unsigned int M10 = 0  # First order moment for j-axis

        cdef unsigned int i, j
        for i in range(0,array.shape[0],sampleStep):
            for j in range(0,array.shape[1],sampleStep):
                if array[i][j] > threshold:
                    if binary == True:
                        M00 += 1
                        M01 += i
                        M10 += j
                    else:
                        M00 += array[i,j]
                        M01 += i * array[i,j]
                        M10 += j * array[i,j]

        cdef double result_i = (1.0 * M01/M00)
        cdef double result_j = (1.0 * M10/M00)
        cdef double[:] result = np.array(
            ([result_i,result_j]),
            dtype='float64')
        return result

    cdef sum_abs_moments_2D(self, unsigned char[:,::1] array,
                            unsigned char threshold, unsigned int sampleStep,
                            double[:] centroid, bint binary=True):
        """Calculates the sum of the magnitude of the array's first order
        moments wrt the centroid of the array. This is similar to the first
        order 'central moments' of the array, but not quite; taking the
        magnitude of the moments (i.e. no negative moments) means that two
        features either side of the centroid don't cancel each other out,
        rather the further away they are, the higher th sum will be.
        Therfore, a low sum indicates a single signal/feature near the
        centroid, whereas multiple signals/features far from the centroid
        will give a higher sum.

        The array is handled by either weighting by the values in the array
        (that are above the threshold), or by giving all elements above the
        threshold equal weight (i.e. binary).

        Args:
            array: A 2D array of values to analyse
            threshold: The value above which elements will be considered
            sampleStep: e.g. 2 for every other element
            centroid: An array of the centroid location ([i,j])
            binary: Flag to give all values above threshold equal weight

        Returns:
            An array of the sum of the magnitude of the array's first order
            central moments:
            [0]: along i-axis
            [1]: along j-axis
        """
        cdef double sum_i = 0  # Sum of absolute moments for i-axis
        cdef double sum_j = 0  # Sum of absolute moments for j-axis

        cdef unsigned int i, j
        for i in range(0,array.shape[0],sampleStep):
            for j in range(0,array.shape[1],sampleStep):
                if array[i][j] > threshold:
                    if binary == True:
                        sum_i += abs(i - centroid[0])
                        sum_j += abs(j - centroid[1])
                    else:
                        sum_i += abs(i - centroid[0]) * array[i,j]
                        sum_j += abs(j - centroid[1]) * array[i,j]

        cdef unsigned int[:] result = np.array(([sum_i,sum_j]), dtype='uint32')
        return result

    cdef xcorr(self, unsigned char[:,::1] array, float fibreDia):
        """Calculates the best coordinates at which to place an ideal fibre
        image (circle) of the expected fibre diameter, by cross-correlating
        the real and ideal images. This currently uses the image_registration
        module's 'chi2_shift' function to find these coordinates.

        The function needs the global variable 'xcorr_array_ideal' to exist,
        which is generated by the 'setup' function.  This must have been run
        with the same fibre diameter passed here.

        Note that the result is returned in terms of image pixel coordinates.

        Args:
            array: A 2D array of values to analyse
            fibreDia: The expected pixel diameter of the circle in the image

        Returns:
            An array of the best place for the centre of the ideal image, in
            image coordinates:
            [0]: along x-axis
            [1]: along y-axis
        """
        cdef double arrayCtr_i = array.shape[0]/2.0
        cdef double arrayCtr_j = array.shape[1]/2.0

        cdef double[:] offsets = np.asarray(
            chi2_shift(
                np.array(array,dtype='uint8'),
                xcorr_array_ideal,
                return_error=False,
                upsample_factor='auto'),
            dtype='float64')

        cdef double result_x = arrayCtr_j - offsets[0]
        cdef double result_y = arrayCtr_i - offsets[1]
        cdef double[:] result = np.array(
            ([result_x,result_y]),
            dtype='float64')
        return result

    cdef biscuit(self, unsigned char[:,::1] array, float fibreDia,
                 unsigned int nth):
        """The 'biscuit cutter' algorithm is similar to the cross-correlation
        method, in that it finds the best sub-pixel position at which a perfect
        circle of a pre-defined radius encircles the highest sum of fibre image
        pixel values. Or in other words, it's like cutting a circular 'biscuit'
        of pixels from the 'dough' of the fibre image.

        This technique uses various tricks to speed up its arrival at a result.
        It uses pre-calculated circular masks (of a diameter matching that of
        the fibre in the image), as well as 'difference masks', which are a
        quick way to repersent moving the mask by a small amount and re-summing
        the image pixels within it.

        The central pixel of the passed array is taken as the staring point for
        finding the centre of the fibre circle. It follows that this pixel must
        be part of or close to the fibre in the image, or the routine could
        scoot off into the darkness and get stuck on a local intensity peak.

        This method needs the masks generated by the 'setup' function in order
        to work. These masks must be square and contain an odd number of
        elements, but the setup function should take care of this. The setup
        must be done with the same fibre diameter as is passed here.

        Args:
            array: A 2D array of values to analyse
            fibreDia: The expected pixel diameter of the circle in the image
            biscuit_nth: The fraction of a pixel to which one wants to measure;
                         i.e. 20 is 1/20 px precision

        Returns:
            An array of the best place for the centre of the ideal image, in
            array coordinates:
            [0]: along i-axis
            [1]: along j-axis
        """
        cdef double[:,::1] fullMask = bisc_fullMask
        cdef int [:,:,:,:,::1] dMask1_coords = bisc_dMask1_coords
        cdef double[:,:,:,::1] dMask1_values = bisc_dMask1_values
        cdef int [:,:,:,:,::1] dMaskSub_coords = bisc_dMaskSub_coords
        cdef double[:,:,:,::1] dMaskSub_values = bisc_dMaskSub_values

        cdef int i, j
        cdef unsigned int k

        cdef unsigned int[:] startPos = np.array(
            ([<unsigned int>(array.shape[0]/2),
              <unsigned int>(array.shape[1]/2)]),
            dtype='uint32')

        cdef unsigned int maskSize_i = fullMask.shape[0]
        cdef unsigned int maskSize_j = fullMask.shape[1]
        cdef unsigned int maskCtr_i = <unsigned int>(bisc_fullMask.shape[0]/2)
        cdef unsigned int maskCtr_j = <unsigned int>(bisc_fullMask.shape[1]/2)

        cdef unsigned int[:] arCtr = startPos
        cdef double score = 0.0

        # Define top-left and bottom-right pixels of the sub-array,
        # onto which the mask is placed:
        cdef unsigned int arTL_i = startPos[0] - maskCtr_i
        cdef unsigned int arTL_j = startPos[1] - maskCtr_j
        cdef unsigned int arBR_i = arTL_i + maskSize_i - 1
        cdef unsigned int arBR_j = arTL_j + maskSize_j - 1

        cdef unsigned char[:,::1] subAr = array[arTL_i:arBR_i+1,
                                                arTL_j:arBR_j+1]

        # Calculate the initial score at the starting position, which is simply
        # the sum of all the pixels under the mask, multiplied by the mask:
        for i in range(subAr.shape[0]):
            for j in range(subAr.shape[1]):
                score += subAr[i,j] * fullMask[i,j]

        cdef double score_max = score

        # Use a single-pixel shift difference mask to find the best whole pixel
        # coordinates of the centre of the fibre circle:
        cdef unsigned int numMaskVals = dMask1_coords.shape[3]
        cdef unsigned char maxTally = 0
        cdef unsigned char d = 0  # Direction number (0=-i, 1=-j, 2=+i, 3=+j)
        cdef bint searching = True  # Is never false (break statements used)
        while searching == True:
            for k in range(numMaskVals):
                i = dMask1_coords[0,0,d,k,0]
                j = dMask1_coords[0,0,d,k,1]

                if (i != -1) and (j != -1):
                    score += dMask1_values[0,0,d,k] * subAr[i,j]
                else:
                    # All the diffmask has been processed
                    break

            if score <= score_max:
                # The current position is better than the trial shift
                score = score_max
                maxTally += 1
                # Increment direction number:
                if d < 3:
                    d += 1
                else:
                    d = 0
            else:
                # The trial shift was better; update sub-image and carry on
                score_max = score
                maxTally = 0

                if d == 0:
                    arTL_i -= 1
                    arBR_i -= 1
                elif d == 1:
                    arTL_j -= 1
                    arBR_j -= 1
                elif d == 2:
                    arTL_i += 1
                    arBR_i += 1
                else:
                    arTL_j += 1
                    arBR_j += 1

                subAr = array[arTL_i:arBR_i+1, arTL_j:arBR_j+1]

            if maxTally == 4:
                # All four possible shifts aren't better, so we're at the max
                break

        # Now use sub-pixel shift difference masks to find the best sub-pixel
        # coordinates of the centre of the fibre circle:
        numMaskVals = dMaskSub_coords.shape[3]
        maxTally = 0
        cdef int i_sub = 0  # Number of nths from centre in i
        cdef int j_sub = 0  # Number of nths from centre in j
        cdef unsigned int i_m, j_m
        cdef bint invert_i, invert_j
        while searching == True:
            for k in range(numMaskVals):
                # Diffmasks are only defined for positive i_sub and j_sub,
                # so masks must be retrieved with positive indices and the
                # direction flipped if necessary. Image coordinates i and j
                # also need mirroring if i_sub or j_sub are negative.
                i_m = abs(i_sub)
                j_m = abs(j_sub)

                if d == 0:  # -i direction
                    if i_sub >= 0:
                        # Normal direction, no coordinate mirroring:
                        d_ = 0
                        invert_i = False
                        invert_j = False
                    else:
                        # Flip direction, mirror i axis:
                        d_ = 2
                        invert_i = True
                        invert_j = False
                elif d == 1:  # -j direction
                    if j_sub >= 0:
                        # Normal direction, no coordinate mirroring:
                        d_ = 1
                        invert_i = False
                        invert_j = False
                    else:
                        # Flip direction, mirror j axis:
                        d_ = 3
                        invert_i = False
                        invert_j = True
                elif d == 2:  # +i direction
                    if i_sub >= 0:
                        # Normal direction, no coordinate mirroring:
                        d_ = 2
                        invert_i = False
                        invert_j = False
                    else:
                        # Flip direction, mirror i axis:
                        d_ = 0
                        invert_i = True
                        invert_j = False
                else:  # +j direction
                    if j_sub >= 0:
                        # Normal direction, no coordinate mirroring:
                        d_ = 3
                        invert_i = False
                        invert_j = False
                    else:
                        # Flip direction, mirror j axis:
                        d_ = 1
                        invert_i = False
                        invert_j = True

                i = dMaskSub_coords[i_m,j_m,d_,k,0]
                j = dMaskSub_coords[i_m,j_m,d_,k,1]

                if (i != -1) and (j != -1):
                    if invert_i == True: i = maskSize_i-1 - i
                    if invert_j == True: j = maskSize_j-1 - j

                    score += dMaskSub_values[i_m,j_m,d_,k] * subAr[i,j]

                else:
                    # All the diffmask has been processed
                    break

            if score <= score_max:
                # The current position is better than the trial shift
                score = score_max
                maxTally += 1
                # Increment direction number:
                if d < 3:
                    d += 1
                else:
                    d = 0
            else:
                # The trial shift was better; update sub-indices and carry on,
                # providing we're not about to go beyond the edge of the pixel
                score_max = score
                maxTally = 0

                if d == 0:
                    i_sub -= 1
                    if i_sub <= -(nth-1): d = 1
                elif d == 1:
                    j_sub -= 1
                    if j_sub <= -(nth-1): d = 2
                elif d == 2:
                    i_sub += 1
                    if i_sub >=  (nth-1): d = 3
                else:
                    j_sub += 1
                    if j_sub >=  (nth-1): d = 0

            if maxTally == 4:
                # All four possible shifts aren't better, so we're at the max
                break

        cdef double arCtr_i = arTL_i + maskCtr_i + (i_sub/<double>nth)
        cdef double arCtr_j = arTL_j + maskCtr_j + (j_sub/<double>nth)
        cdef double[:] result = np.array(([arCtr_i,arCtr_j]), dtype='float64')
        return result

    cdef fitparab(self, unsigned char[:,::1] array, float fibreDia):
        """Calculates the coordinates of the vertices of parabolas fitted to
        1D projections of a 2D array (least squares).

        The expected fibre diameter is used to set a value threshold for the
        array, below which pixels are not considered in the fit. The threshold
        is simply the pixel value represented by the appropriate percentile of
        array elements (pixels). This does a good job of cutting off the
        'wings' of the profile caused by optical effects, leaving only the
        parabolic profile of the fibre core.

        Args:
            array: A 2D array of values to analyse
            fibreDia: The expected pixel diameter of the circle in the image

        Returns:
            An array of the parabola vertex, in array coordinates:
            [0]: along i-axis
            [1]: along j-axis
        """
        array_proj_i = np.zeros(shape=(array.shape[0]))
        array_proj_j = np.zeros(shape=(array.shape[1]))

        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                val = array[i,j]
                array_proj_i[i] += val
                array_proj_j[j] += val

        px_i = np.arange(array_proj_i.shape[0])
        px_j = np.arange(array_proj_j.shape[0])

        fitThresh_i = np.percentile(
            array_proj_i,
            100 * fibreDia/array_proj_i.shape[0])
        fitThresh_j = np.percentile(
            array_proj_j,
            100 * fibreDia/array_proj_j.shape[0])

        pfit_i = np.polyfit(
            px_i, array_proj_i, 2,
            w=(array_proj_i>fitThresh_i))
        pfit_j = np.polyfit(
            px_j, array_proj_j, 2,
            w=(array_proj_j>fitThresh_j))
        
        cdef double result_i = (-1*pfit_i[1])/(2*pfit_i[0])  # Parabola vertex
        cdef double result_j = (-1*pfit_j[1])/(2*pfit_j[0])
        cdef double[:] result = np.array(
            ([result_i,result_j]),
            dtype='float64')
        return result


    def __dealloc__(self):
        pass


class FibreImageError(Exception):
    pass

