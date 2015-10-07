"""Makes various types of mask for the 'biscuit cutter' circular centre-finding
routine. These aren't intended to be called directly, rather by a higher-level
method that knows what arguments should be passed for various circle radii etc.

Everything works in array coordinates ([i,j]) unless specified. Also note that
the terms 'array element' and 'pixel' can be, and are, used interchangeably.
"""

import numpy as np
import circlemaker


def make_mask(circR, ctrOffset):
    """Generates a 'mask' array, which contains values from 0.0 to 1.0, in the
    shape of a spot, where 1.0 is inside the spot, 0.0 is outside, and middle
    values are accurate fractional representations of the array elements at the
    edge of the circle. Any radius of circle can be specified, as well as an
    offset of 0 to 1 pixels for the centre of the circle. The mask size will be
    the circle diameter plus 1 empty pixel in both i and j, hence a maximum
    offset of 1 pixel for the centre coordinates. This also means the returned
    array will always be square and have an odd number of elements.

    Args:
        circR: The radius of the circle in the mask, in pixels (floats ok)
        ctrOffset: The offset of the centre of the circle from the actual
                   centre of the array, between 0.0 and 1.0 pixels, in array
                   element coordinates ([i,j]). Sub-element values are fine.

    Returns:
        A numpy array of mask values from 0.0 to 1.0
    """
    ctrOffset = np.asarray(ctrOffset, dtype='float64')

    maskSize_ij = int((np.ceil(circR+0.5)*2) + 1)
    maskSize = np.array([maskSize_ij, maskSize_ij])
    circCtr_i = (maskSize[0]/2.0) + ctrOffset[0]
    circCtr_j = (maskSize[1]/2.0) + ctrOffset[1]
    circCtr = np.array([circCtr_j, circCtr_i])  # Circlemaker uses image coords
    mask = circlemaker.make_circle(maskSize, circR, circCtr, 1.0)

    return np.asarray(mask, dtype='float64')

def make_diffmask(circR, ctrOffset, shift):
    """Generates a single circular 'difference mask', which is simply the
    result of subtracting a normal mask from a shifted version of itself.
    A shift of 0.0 to 1.0 is expected, in i and/or j.

    Args:
        circR: The radius of the circle in the mask, in pixels (floats ok)
        ctrOffset: The offset of the centre of the reference circle from the
                   centre of the array, between 0.0 and 1.0 pixels, in array
                   element coordinates ([i,j]). Sub-element values are fine.
        shift: The number of pixels to shift the circle by, wrt 'ctrOffset',
               in array element coordinates ([i,j]). Sub-element values ok.

    Returns:
        A numpy array representing the difference between shifted and unshifted
        versions of a whole mask
    """
    ctrOffset = np.asarray(ctrOffset, dtype='float64')
    shift = np.asarray(shift, dtype='float64')

    mask_unshifted = make_mask(circR, ctrOffset)
    mask_shifted   = make_mask(circR, ctrOffset+shift)
    
    mask_difference = mask_shifted - mask_unshifted
    
    return np.asarray(mask_difference, dtype='float64')

def make_many_diffmasks(circR, nth):
    """Generates a (compressed) catalogue of difference masks. This can get a
    bit confusing, and unfortunately it is best explained with a picture. In a
    nutshell, the returned arrays contain all the information needed to
    represent shifts of an nth of a pixel is any direction, from any location
    in the sub-divided element or pixel of the image being processed.

    e.g. With nth=10, the contents of the difference mask representing a 10th
    of a pixel shift in the +i direction, from the coordinates (0.8,0.2), and
    hence *to* coordinates (0.9,0.2), is at maskData[8,2,2].

    The returned data is a 'compressed' representation of all the difference
    masks. This is worth doing because most difference masks contain zeros,
    which do nothing but take up memory. So instead of the returned data
    containing every difference mask in its raw form, it contains only the i
    coordinate, the j coordinate, and the actual value at those coordinates;
    all zeros are ignored. This is actually returned as two different arrays
    because of data types (see 'Returns' below).

    To further save memory, symmetry is exploited and only positive sub-element
    shifts are considered. Negative sub-element coordinates have masks just
    like their positive reflections, only the difference mask coordinates need
    to be inverted as necessary.

    e.g. With nth=10, the difference mask representing a 10th of a pixel shift
    in the -j direction, from the coordinates (-0.1,0.0), and hence *to*
    coordinates (-0.1,-0.1), is at maskData[1,0,3]. Note the necessary flipping
    of the direction number. The i coordinates of the mask will also need to be
    mirrored.

    Args:
        circR: The radius of the circle in the mask, in pixels (floats ok)
        nth: How many sub-elements a pixel is divided into; i.e. 20 -> 1/20

    Returns:
        A tuple of numpy arrays, (a,b), where:
        a:  A numpy array of all the i and j coordinates where non-zero values
            exist, for all difference masks. Indexing is as follows. e.g. all
            the i,j pairs for a +i shift from image sub-pixel coordinates
            (0.8,0.1), with nth=10, would be found at [8,1,2,:,:].
                Index 0: How many pixel sub-divisions in the i direction
                Index 1: How many pixel sub-divisions in the j direction
                Index 3: The direction of the shift ([-i,-j,+i,+j])
        b:  The same indexing as 'a', but containing the actual mask values
            found at the corresponding coordinates. e.g. all the values for a
            +i shift from image sub-pixel coordinates (0.8,0.1), with nth=10,
            would be found at [8,1,2,:].
    """
    maskSize = int((np.ceil(circR+0.5)*2) + 1)  # Must match make_mask output
    nth_ = float(nth)  # Precise float version for fractional calculations

    manyMasks = np.zeros(shape=(nth,nth,4,maskSize,maskSize), dtype='float64')

    # All masks are stored in full at first, as this is the only way to know
    # the largest number of non-zero elements across all masks (maxNon0):
    maxNon0 = 0
    for i in range(nth):
        for j in range(nth):
            manyMasks[i,j,0] = make_diffmask(circR, (i/nth_,j/nth_),
                                             (-1.0/nth_, 0.0) )
            manyMasks[i,j,1] = make_diffmask(circR, (i/nth_,j/nth_),
                                             (0.0, -1.0/nth_) )
            manyMasks[i,j,2] = make_diffmask(circR, (i/nth_,j/nth_),
                                             (1.0/nth_, 0.0) )
            manyMasks[i,j,3] = make_diffmask(circR, (i/nth_,j/nth_),
                                             (0.0, 1.0/nth_) )

            maxNon0 = np.max([(manyMasks[i,j,0] != 0).sum(), maxNon0])
            maxNon0 = np.max([(manyMasks[i,j,1] != 0).sum(), maxNon0])
            maxNon0 = np.max([(manyMasks[i,j,2] != 0).sum(), maxNon0])
            maxNon0 = np.max([(manyMasks[i,j,3] != 0).sum(), maxNon0])

    # Now to compress the monstrous array of arrays we just calculated
    # The following code essentially just goes through them all, ignoring zeros
    # If there are fewer non-zero values than in other masks, we pad with -1
    maskData_coords = np.zeros(shape=(nth,nth,4,maxNon0,2), dtype='int32'  )
    maskData_vals   = np.zeros(shape=(nth,nth,4,maxNon0),   dtype='float64')
    for i in range(nth):
        for j in range(nth):
            for d in range(4):
                # First we find the coordinates of non-zero values:
                (coords_i, coords_j) = np.where(manyMasks[i,j,d] != 0)
                padSize = maxNon0 - coords_i.shape[0]
                maskData_coords[i,j,d,:,0] = np.pad(coords_i,
                                                    (0,padSize),
                                                    'constant',
                                                    constant_values=(-1) )
                maskData_coords[i,j,d,:,1] = np.pad(coords_j,
                                                    (0,padSize),
                                                    'constant',
                                                    constant_values=(-1) )
                # Now find the values at those coordinates: 
                vals = manyMasks[i,j,d][manyMasks[i,j,d] != 0]
                padSize = maxNon0 - vals.shape[0]
                maskData_vals[i,j,d,:] = np.pad(vals,
                                                (0,padSize),
                                                'constant',
                                                constant_values=0 )

    return (maskData_coords, maskData_vals)


