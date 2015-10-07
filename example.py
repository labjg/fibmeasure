import numpy as np
from scipy import misc
from fibmeasure import measure


def print_result(result, ref):
    """A little function to neatly print measurement results"""
    print("Estimated fibre centre:  (%.3f, %.3f) wrt refPoint\n"\
          "                         (%.3f, %.3f) absolute"
          % (result[0], result[1], result[0]+ref[0], result[1]+ref[1]))


# Run initial setup routine for the fibmeasure methods. We know the theoretical
# fibre diameter in pixels from the camera's optical design and calibration.
# nth=20 means that the 'biscuit cutter' routine will have 1/20 px precision:
measure.setup(fibreDia=26.50, saveDir="fibmeasure_setup_data/",
              setup_xcorr=True, setup_biscuit=True, biscuit_nth=20)

# Load some sample images into numpy arrays:
im_single = misc.imread("test_images/im_single.tif")
im_single_array = np.array(im_single, dtype='uint8')
del im_single
im_multi = misc.imread("test_images/im_multi.tif")
im_multi_array = np.array(im_multi, dtype='uint8')
del im_multi


# First, do some example measurements for a single-fibre image:
refPoint = np.array(([640.0,480.0]), dtype='float64')
fibim = measure.FibreImage(im_single_array, refPoint)
print("\nSingle-fibre sample image with reference point (%.3f, %.3f)..."
      % (refPoint[0], refPoint[1]))

print("\nCentroid method...")
ctr_centroid = fibim.get_centre_centroid(fibreDia=26.50)
print_result(ctr_centroid, refPoint)

print("\nCross-correlation method...")
ctr_xcorr = fibim.get_centre_xcorr()
print_result(ctr_xcorr, refPoint)

print("\nBiscuit cutter method...")
ctr_biscuit = fibim.get_centre_biscuit()
print_result(ctr_biscuit, refPoint)

print("\nParabolic fit method...")
ctr_fitparab = fibim.get_centre_fitparab(fibreDia=26.50)
print_result(ctr_fitparab, refPoint)


# Now do some example measurements for a multiple-fibre image:
refPoint = np.array(([640.0,480.0]), dtype='float64')
fibim = measure.FibreImage(im_multi_array, refPoint)
print("\nMulti-fibre sample image with reference point (%.3f, %.3f)..."
      % (refPoint[0], refPoint[1]))

print("\nCentroid method...")
ctr_centroid = fibim.get_centre_centroid(fibreDia=26.50)
print_result(ctr_centroid, refPoint)

print("\nBiscuit cutter method...")
ctr_biscuit = fibim.get_centre_biscuit()
print_result(ctr_biscuit, refPoint)


# Now change the reference point, to change which fibre is selected:
refPoint = np.array(([970.0,440.0]), dtype='float64')
fibim = measure.FibreImage(im_multi_array, refPoint)
print("\nMulti-fibre sample image with reference point (%.3f, %.3f)..."
      % (refPoint[0], refPoint[1]))

print("\nCentroid method...")
ctr_centroid = fibim.get_centre_centroid(fibreDia=26.50)
print_result(ctr_centroid, refPoint)

print("\nBiscuit cutter method...")
ctr_biscuit = fibim.get_centre_biscuit()
print_result(ctr_biscuit, refPoint)


