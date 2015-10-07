"""This is a horrid little module, written by a boy rather new to Python, and
documented since by the same boy, now cringing. It seems to work though.

Only the first function ('make_circle') is meant to be called directly, and it
returns an image array with a filled circle (spot) in it, of a specified radius
and with a specified centre coordinate. The important thing is that these
values are of high precision. The returned array accurately represents the edge
of the circle too, which is nice.

Watch out for the mixing of coordinate systems. Sometimes you'll have image
coordinates (x,y), where (0,0) is the top-left corner of the top-left pixel in
the image, and sometimes you'll have array coordinates (i,j), row number first,
where (0,0) is the top-left _element_ in the array.
"""

import numpy as np


def make_circle(canvasSize, CircR, CircCtr, fill):
    """Makes an image array of a certain size, containing a precisely-drawn
    filled circle of a given radius, with a given centre point. The pixel
    values in the array range from zero (outside the circle) to the specified
    'fill' value, with pixels around the edge of the circle representing the
    fraction of that pixel that is bounded by the circle.

    The procedures below use numerous lists, which are unreasonably difficult
    to decode. There is also the concept of north, south, east and west sides
    of a pixel, which in real terms are the sides in the -i, +i, +j and -j
    directions respectively. There's also the 'class' system, which has nothing
    to do with object classes or aristocracy; each pixel is assigned a class
    number based on where the theoretical perfect circle intersects its
    boundaries.

    Args:
        canvasSize: The size of the returned (2D) image (x,y)
        CircR: The radius of the filled circle to be drawn (floats are fine)
        CircCtr: The centre point of the circle, in image coordinates (x,y)
        fill: The value of a pixel completely within the circle (floats are ok)

    Returns:
        A 2D numpy array of floating point values, where pixels inside the
        circle have the value 'fill' and those outside have the value 0. Pixels
        at the edge of the circle are of an appropriate intermediate value.
    """
    # We begin with a hack to avoid the circle exactly touching a pixel border:
    CircR -= 0.00000000001

    circleArr = np.zeros((canvasSize[1],canvasSize[0]), dtype='float64')

    #DEBUG:
    #Classcircle = np.zeros((canvasSize[1],canvasSize[0]))

    CrossingsX = []  # For storing the crossings for all x and y values
    CrossingsY = []

    # Now create the lists of line crossings; e.g. the 5th element of the
    # 'CrossingsX' array will contain a tuple of the y co-ords where the circle
    # crosses the line 'x=5', which divides the 5th and 6th column of pixels:
    for i in range(0,(np.max([canvasSize[0],canvasSize[1]]))+1):
        CrossingsX.extend(solve_circ_y(CircR,CircCtr[0],CircCtr[1],i))
        CrossingsY.extend(solve_circ_x(CircR,CircCtr[0],CircCtr[1],i))
    
    # Now visit each pixel and look at its four 'borders' with its neighbours;
    # is there a line crossing through any of the borders?
    for x in range(0,canvasSize[0]):
        for y in range(0,canvasSize[1]):
            Point1 = ((-1,-1))
            Point2 = ((-1,-1))
            Point3 = ((-1,-1))
            Point4 = ((-1,-1))
            BoundsX = ((x,x+1))
            BoundsY = ((y,y+1))
            pxClass = 0
            
            # Do not change the following direction order (i.e. N,S,E,W), or
            # class 11,13,17,14 area calculations will be incorrect.
            
            # North border:
            if (BoundsX[0] <= CrossingsY[BoundsY[0]][0] <= BoundsX[1]) or (BoundsX[0] <= CrossingsY[BoundsY[0]][1] <= BoundsX[1]):  #if there is any crossing along Y = (north border)
                if BoundsX[0] <= CrossingsY[BoundsY[0]][0] <= BoundsX[1]:  #does crossing A lie within this pixel?
                    if Point1[0] < 0:
                        Point1 = (( (float(CrossingsY[BoundsY[0]][0])), float((BoundsY[0])) ))
                    elif Point2[0] < 0:
                        Point2 = (( (float(CrossingsY[BoundsY[0]][0])), float((BoundsY[0])) ))
                    elif Point3[0] < 0:
                        Point3 = (( (float(CrossingsY[BoundsY[0]][0])), float((BoundsY[0])) ))
                    else:
                        Point4 = (( (float(CrossingsY[BoundsY[0]][0])), float((BoundsY[0])) ))
                if BoundsX[0] <= CrossingsY[BoundsY[0]][1] <= BoundsX[1]:  #does crossing B lie within this pixel?
                    if Point1[0] < 0:
                        Point1 = (( (float(CrossingsY[BoundsY[0]][1])), float((BoundsY[0])) ))
                    elif Point2[0] < 0:
                        Point2 = (( (float(CrossingsY[BoundsY[0]][1])), float((BoundsY[0])) ))
                    elif Point3[0] < 0:
                        Point3 = (( (float(CrossingsY[BoundsY[0]][1])), float((BoundsY[0])) ))
                    else:
                        Point4 = (( (float(CrossingsY[BoundsY[0]][1])), float((BoundsY[0])) ))
                pxClass += 1
                    
            # South border:
            if (BoundsX[0] <= CrossingsY[BoundsY[1]][0] <= BoundsX[1]) or (BoundsX[0] <= CrossingsY[BoundsY[1]][1] <= BoundsX[1]):
                if BoundsX[0] <= CrossingsY[BoundsY[1]][0] <= BoundsX[1]:
                    if Point1[0] < 0:
                        Point1 = (( (float(CrossingsY[BoundsY[1]][0])), float((BoundsY[1])) ))
                    elif Point2[0] < 0:
                        Point2 = (( (float(CrossingsY[BoundsY[1]][0])), float((BoundsY[1])) ))
                    elif Point3[0] < 0:
                        Point3 = (( (float(CrossingsY[BoundsY[1]][0])), float((BoundsY[1])) ))
                    else:
                        Point4 = (( (float(CrossingsY[BoundsY[1]][0])), float((BoundsY[1])) ))
                if BoundsX[0] <= CrossingsY[BoundsY[1]][1] <= BoundsX[1]:
                    if Point1[0] < 0:
                        Point1 = (( (float(CrossingsY[BoundsY[1]][1])), float((BoundsY[1])) ))
                    elif Point2[0] < 0:
                        Point2 = (( (float(CrossingsY[BoundsY[1]][1])), float((BoundsY[1])) ))
                    elif Point3[0] < 0:
                        Point3 = (( (float(CrossingsY[BoundsY[1]][1])), float((BoundsY[1])) ))
                    else:
                        Point4 = (( (float(CrossingsY[BoundsY[1]][1])), float((BoundsY[1])) ))
                pxClass += 2
        
            # East border:
            if (BoundsY[0] < CrossingsX[BoundsX[1]][0] < BoundsY[1]) or (BoundsY[0] < CrossingsX[BoundsX[1]][1] < BoundsY[1]):
                if BoundsY[0] < CrossingsX[BoundsX[1]][0] < BoundsY[1]:
                    if Point1[0] < 0:
                        Point1 = (( float((BoundsX[1])), (float(CrossingsX[BoundsX[1]][0])) ))
                    elif Point2[0] < 0:
                        Point2 = (( float((BoundsX[1])), (float(CrossingsX[BoundsX[1]][0])) ))
                    elif Point3[0] < 0:
                        Point3 = (( float((BoundsX[1])), (float(CrossingsX[BoundsX[1]][0])) ))
                    else:
                        Point4 = (( float((BoundsX[1])), (float(CrossingsX[BoundsX[1]][0])) ))
                if BoundsY[0] < CrossingsX[BoundsX[1]][1] < BoundsY[1]:
                    if Point1[0] < 0:
                        Point1 = (( float((BoundsX[1])), (float(CrossingsX[BoundsX[1]][1])) ))
                    elif Point2[0] < 0:
                        Point2 = (( float((BoundsX[1])), (float(CrossingsX[BoundsX[1]][1])) ))
                    elif Point3[0] < 0:
                        Point3 = (( float((BoundsX[1])), (float(CrossingsX[BoundsX[1]][1])) ))
                    else:
                        Point4 = (( float((BoundsX[1])), (float(CrossingsX[BoundsX[1]][1])) ))
                pxClass += 4
        
            # West border:
            if (BoundsY[0] < CrossingsX[BoundsX[0]][0] < BoundsY[1]) or (BoundsY[0] < CrossingsX[BoundsX[0]][1] < BoundsY[1]):
                if BoundsY[0] < CrossingsX[BoundsX[0]][0] < BoundsY[1]:
                    if Point1[0] < 0:
                        Point1 = (( float((BoundsX[0])), (float(CrossingsX[BoundsX[0]][0])) ))
                    elif Point2[0] < 0:
                        Point2 = (( float((BoundsX[0])), (float(CrossingsX[BoundsX[0]][0])) ))
                    elif Point3[0] < 0:
                        Point3 = (( float((BoundsX[0])), (float(CrossingsX[BoundsX[0]][0])) ))
                    else:
                        Point4 = (( float((BoundsX[0])), (float(CrossingsX[BoundsX[0]][0])) ))
                if BoundsY[0] < CrossingsX[BoundsX[0]][1] < BoundsY[1]:
                    if Point1[0] < 0:
                        Point1 = (( float((BoundsX[0])), (float(CrossingsX[BoundsX[0]][1])) ))
                    elif Point2[0] < 0:
                        Point2 = (( float((BoundsX[0])), (float(CrossingsX[BoundsX[0]][1])) ))
                    elif Point3[0] < 0:
                        Point3 = (( float((BoundsX[0])), (float(CrossingsX[BoundsX[0]][1])) ))
                    else:
                        Point4 = (( float((BoundsX[0])), (float(CrossingsX[BoundsX[0]][1])) ))
                pxClass += 8
            
            # The pixel now has a unique 'class number' which encodes which pair of borders contain the crossing with the circle; the encoding is via binary factors: 1 for North wall, 2 for South, 4 for East, 8 for West; 16 is added if the pixel is an 'inverse' type (see below)
            if pxClass==0:  #if the circle doesnt touch pixel (i.e. no crossings)
                if np.sqrt( (CircCtr[0]-x)**2 + (CircCtr[1]-y)**2 ) < CircR:  #if the pixel is inside the circle
                    circleArr[y][x] = 1.0
                else:  #else outside the circle
                    circleArr[y][x] = 0.0
        
            elif pxClass==6:  #if south-east intersections
                if CircCtr[0]>(x+0.5) and CircCtr[1]>(y+0.5):  #if 'normal' (minor area enclosed)
                    ASeg = circ_seg(Point1[0],Point1[1],Point2[0],Point2[1],CircR)
                    ATri = triangle_area(Point1[0],Point1[1],Point2[0],Point2[1])
                    circleArr[y][x] = ASeg+ATri
                else:  #else 'inverse' (major area enclosed)
                    pxClass += 16
                    ASeg = circ_seg(Point1[0],Point1[1],Point2[0],Point2[1],CircR)
                    ATri = triangle_area(Point1[0],Point1[1],Point2[0],Point2[1])
                    circleArr[y][x] = ASeg+(1-ATri)
            elif pxClass==10:  #if south-west intersections
                if CircCtr[0]<(x+0.5) and CircCtr[1]>(y+0.5):  #if 'normal' (minor area enclosed)
                    ASeg = circ_seg(Point1[0],Point1[1],Point2[0],Point2[1],CircR)
                    ATri = triangle_area(Point1[0],Point1[1],Point2[0],Point2[1])
                    circleArr[y][x] = ASeg+ATri
                else:  #else 'inverse' (major area enclosed)
                    pxClass += 16
                    ASeg = circ_seg(Point1[0],Point1[1],Point2[0],Point2[1],CircR)
                    ATri = triangle_area(Point1[0],Point1[1],Point2[0],Point2[1])
                    circleArr[y][x] = ASeg+(1-ATri)
            elif pxClass==5:  #if north-east intersections
                if CircCtr[0]>(x+0.5) and CircCtr[1]<(y+0.5):  #if 'normal' (minor area enclosed)
                    ASeg = circ_seg(Point1[0],Point1[1],Point2[0],Point2[1],CircR)
                    ATri = triangle_area(Point1[0],Point1[1],Point2[0],Point2[1])
                    circleArr[y][x] = ASeg+ATri
                else:  #else 'inverse' (major area enclosed)
                    pxClass += 16
                    ASeg = circ_seg(Point1[0],Point1[1],Point2[0],Point2[1],CircR)
                    ATri = triangle_area(Point1[0],Point1[1],Point2[0],Point2[1])
                    circleArr[y][x] = ASeg+(1-ATri)
            elif pxClass==9:  #if north-west intersections
                if CircCtr[0]<(x+0.5) and CircCtr[1]<(y+0.5):  #if 'normal' (minor area enclosed)
                    ASeg = circ_seg(Point1[0],Point1[1],Point2[0],Point2[1],CircR)
                    ATri = triangle_area(Point1[0],Point1[1],Point2[0],Point2[1])
                    circleArr[y][x] = ASeg+ATri
                else:  #else 'inverse' (major area enclosed)
                    pxClass += 16
                    ASeg = circ_seg(Point1[0],Point1[1],Point2[0],Point2[1],CircR)
                    ATri = triangle_area(Point1[0],Point1[1],Point2[0],Point2[1])
                    circleArr[y][x] = ASeg+(1-ATri)
        
            elif pxClass==12:  #if east-west intersections
                if CircCtr[1]<(y+0.5):  #if 'normal' (north side enclosed)
                    ASeg = circ_seg(Point1[0],Point1[1],Point2[0],Point2[1],CircR)
                    ATri = triangle_area(Point1[0],Point1[1],Point2[0],Point2[1])
                    ARec = rectangle_area(Point1[1],Point2[1])
                    circleArr[y][x] = ASeg+ATri+ARec
                else:  #else 'inverse' (south side enclosed)
                    pxClass += 16
                    ASeg = circ_seg(Point1[0],Point1[1],Point2[0],Point2[1],CircR)
                    ATri = triangle_area(Point1[0],Point1[1],Point2[0],Point2[1])
                    ARec = rectangle_area(Point1[1],Point2[1])
                    circleArr[y][x] = ASeg+(1-(ATri+ARec))
            elif pxClass==3:  #if north-south intersections
                if CircCtr[0]<(x+0.5):  #if 'normal' (west side enclosed)
                    ASeg = circ_seg(Point1[0],Point1[1],Point2[0],Point2[1],CircR)
                    ATri = triangle_area(Point1[0],Point1[1],Point2[0],Point2[1])
                    ARec = rectangle_area(Point1[0],Point2[0])
                    circleArr[y][x] = ASeg+ATri+ARec
                else:  #else 'inverse' (east side enclosed)
                    pxClass += 16
                    ASeg = circ_seg(Point1[0],Point1[1],Point2[0],Point2[1],CircR)
                    ATri = triangle_area(Point1[0],Point1[1],Point2[0],Point2[1])
                    ARec = rectangle_area(Point1[0],Point2[0])
                    circleArr[y][x] = ASeg+(1-(ATri+ARec))
            
            elif pxClass==11:  #if north-west AND south-west intersections
                PointA = Point1
                PointD = Point2
                if Point3[1]<Point4[1]:
                    PointB = Point3
                    PointC = Point4
                else:
                    PointB = Point4
                    PointC = Point3
                ASeg1 = circ_seg(PointA[0],PointA[1],PointB[0],PointB[1],CircR)
                ATri1 = triangle_area(PointA[0],PointA[1],PointB[0],PointB[1])
                ASeg2 = circ_seg(PointC[0],PointC[1],PointD[0],PointD[1],CircR)
                ATri2 = triangle_area(PointC[0],PointC[1],PointD[0],PointD[1])
                circleArr[y][x] = ASeg1+ASeg2+(1-ATri1-ATri2)
            elif pxClass==7:  #if north-east AND south-east intersections
                PointA = Point1
                PointD = Point2
                if Point3[1]<Point4[1]:
                    PointB = Point3
                    PointC = Point4
                else:
                    PointB = Point4
                    PointC = Point3
                ASeg1 = circ_seg(PointA[0],PointA[1],PointB[0],PointB[1],CircR)
                ATri1 = triangle_area(PointA[0],PointA[1],PointB[0],PointB[1])
                ASeg2 = circ_seg(PointC[0],PointC[1],PointD[0],PointD[1],CircR)
                ATri2 = triangle_area(PointC[0],PointC[1],PointD[0],PointD[1])
                circleArr[y][x] = ASeg1+ASeg2+(1-ATri1-ATri2)
            elif pxClass==13:  #if north-west AND north-east intersections
                PointA = Point4
                PointD = Point3
                if Point1[0]<Point2[0]:
                    PointB = Point1
                    PointC = Point2
                else:
                    PointB = Point1
                    PointC = Point2
                ASeg1 = circ_seg(PointA[0],PointA[1],PointB[0],PointB[1],CircR)
                ATri1 = triangle_area(PointA[0],PointA[1],PointB[0],PointB[1])
                ASeg2 = circ_seg(PointC[0],PointC[1],PointD[0],PointD[1],CircR)
                ATri2 = triangle_area(PointC[0],PointC[1],PointD[0],PointD[1])
                circleArr[y][x] = ASeg1+ASeg2+(1-ATri1-ATri2)
            elif pxClass==14:  #if south-west AND south-east intersections
                PointA = Point4
                PointD = Point3
                if Point1[0]<Point2[0]:
                    PointB = Point1
                    PointC = Point2
                else:
                    PointB = Point1
                    PointC = Point2
                ASeg1 = circ_seg(PointA[0],PointA[1],PointB[0],PointB[1],CircR)
                ATri1 = triangle_area(PointA[0],PointA[1],PointB[0],PointB[1])
                ASeg2 = circ_seg(PointC[0],PointC[1],PointD[0],PointD[1],CircR)
                ATri2 = triangle_area(PointC[0],PointC[1],PointD[0],PointD[1])
                circleArr[y][x] = ASeg1+ASeg2+(1-ATri1-ATri2)
            
            elif pxClass==1:  #if both intersections on north border (just a circular segment area)
                ASeg = circ_seg(Point1[0],Point1[1],Point2[0],Point2[1],CircR)
                circleArr[y][x] = ASeg
            elif pxClass==2:  #if both intersections on south border (just a circular segment area)
                ASeg = circ_seg(Point1[0],Point1[1],Point2[0],Point2[1],CircR)
                circleArr[y][x] = ASeg
            elif pxClass==4:  #if both intersections on east border (just a circular segment area)
                ASeg = circ_seg(Point1[0],Point1[1],Point2[0],Point2[1],CircR)
                circleArr[y][x] = ASeg
            elif pxClass==8:  #if both intersections on west border (just a circular segment area)
                ASeg = circ_seg(Point1[0],Point1[1],Point2[0],Point2[1],CircR)
                circleArr[y][x] = ASeg
               
            #DEBUG:
            #pxClasscircle[y][x] = pxClass
    
    if fill != 1.0:
        for x in range(0,canvasSize[0]):
            for y in range(0,canvasSize[1]):
                circleArr[y][x] = circleArr[y][x] * fill
    
    return circleArr


# The following functions are not intended to be called directly

def solve_circ_y(r, a, b, x):
    """Find y of intersection of circle with x=something"""
    # (x-a)^2 + (y-b)^2 = r^2, where a and b are centre co-ordinates
    # So: y^2 - 2by + b^2 + (x-a)^2 - r^2 = 0
    # Calculate quadratic coefficients and get roots:
    quad_a = 1.0
    quad_b = -2 * b
    quad_c = b**2 + ((x-a)**2) - r**2
    return [ quad_roots(quad_a,quad_b,quad_c) ]

def solve_circ_x(r, a, b, y):
    """Find x of intersection of circle with y=something"""
    # (x-a)^2 + (y-b)^2 = r^2, where a and b are centre co-ordinates
    # So: x^2 - 2ax + a^2 + (y-b)^2 - r^2 = 0
    # Calculate quadratic coefficients and get roots:
    quad_a = 1.0
    quad_b = -2*a
    quad_c = a**2 + ((y-b)**2) - r**2
    return [ quad_roots(quad_a,quad_b,quad_c) ]

def quad_roots(a, b, c):
    """A slightly hacked function to find the roots of a quadratic equation"""
    discriminant = b**2 - 4*a*c
    if discriminant > 0:
        root = np.sqrt(discriminant)
        return [ (-b + root) / (2*a), (-b - root) / (2*a) ]
    else:
        return [-1,-1]

def euclidean(x1, y1, x2, y2):
    """Calculate euclidean distance between two points (x1,y1) and (x2,y2)"""
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)

def circ_seg(x1, y1, x2, y2, r):
    """Calculate area bound by circle and chord (circular segment area)"""
    x = euclidean(x1,y1,x2,y2) / 2
    y = np.sqrt(r**2 - x**2)
    ATriangles = x*y
    theta = 2*(np.arcsin(x/r))  #angle of sector (radians)
    ACircle = np.pi*(r**2)  #area of whole circle
    ASector = ACircle * (theta/(2*np.pi))  #area of sector
    A = ASector - ATriangles  #final area of segment
    return A

def triangle_area(x1, y1, x2, y2):
    """Calculate the area of the triangle bound by the chord and the pixel walls."""
    return np.abs((x2-x1)*(y2-y1)) * 0.5

def rectangle_area(a, b):
    """Calculate the area of the rectangle within a pixel, up to the first crossing point."""
    return np.min([a,b]) - int(np.min([a,b]))

