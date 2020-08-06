"""
Contains various functions for computing statistics over 3D volumes
"""
import numpy as np

def Dice3d(a, b):
    """
    This will compute the Dice Similarity coefficient for two 3-dimensional volumes
    Volumes are expected to be of the same size. We are expecting binary masks -
    0's are treated as background and anything else is counted as data

    Arguments:
        a {Numpy array} -- 3D array with first volume
        b {Numpy array} -- 3D array with second volume

    Returns:
        float
    """
    if len(a.shape) != 3 or len(b.shape) != 3:
        raise Exception(f"Expecting 3 dimensional inputs, got {a.shape} and {b.shape}")

    if a.shape != b.shape:
        raise Exception(f"Expecting inputs of the same shape, got {a.shape} and {b.shape}")

    # Write implementation of Dice3D. If you completed exercises in the lessons
    # you should already have it.
    
    intersection = np.sum((a > 0) * (b > 0))
    sum_volumes = np.sum(a > 0) + np.sum(b > 0)
    if sum_volumes == 0:
        return -1
    return 2.0 * float(intersection) / float(sum_volumes)

def Jaccard3d(a, b):
    """
    This will compute the Jaccard Similarity coefficient for two 3-dimensional volumes
    Volumes are expected to be of the same size. We are expecting binary masks - 
    0's are treated as background and anything else is counted as data

    Arguments:
        a {Numpy array} -- 3D array with first volume
        b {Numpy array} -- 3D array with second volume

    Returns:
        float
    """
    if len(a.shape) != 3 or len(b.shape) != 3:
        raise Exception(f"Expecting 3 dimensional inputs, got {a.shape} and {b.shape}")

    if a.shape != b.shape:
        raise Exception(f"Expecting inputs of the same shape, got {a.shape} and {b.shape}")

    # Write implementation of Jaccard similarity coefficient. Please do not use 
    # the Dice3D function from above to do the computation ;)
    intersection = np.sum((a > 0) * (b > 0))
    sum_volumes = np.sum(a > 0) + np.sum(b > 0) - intersection
    if sum_volumes == 0:
        return -1
    return float(intersection) / float(sum_volumes)

def Spec3d(a, b):
    '''
    This will compute the Sensitivity for two 3-dimensional volumes
    Volumes are expected to be of the same size. We are expecting binary masks - 
    0's are treated as background and anything else is counted as data

    Arguments:
        a {Numpy array} -- prediction 3D array with first volume
        b {Numpy array} -- ground truth 3D array with second volume

    Returns:
        float
    '''
    
    if len(a.shape) != 3 or len(b.shape) != 3:
        raise Exception(f"Expecting 3 dimensional inputs, got {a.shape} and {b.shape}")

    if a.shape != b.shape:
        raise Exception(f"Expecting inputs of the same shape, got {a.shape} and {b.shape}")

    tn = float(np.sum((b==0)*(a==0)))
    fp = float(np.sum((b==0)*(a > 0)))

    if (tn + fp) == 0:
        return -1
    return tn / (tn + fp)

def Sens3d(a, b):
    '''
    This will compute the Sensitivity for two 3-dimensional volumes
    Volumes are expected to be of the same size. We are expecting binary masks - 
    0's are treated as background and anything else is counted as data

    Arguments:
        a {Numpy array} -- prediction 3D array with first volume
        b {Numpy array} -- ground truth 3D array with second volume

    Returns:
        float
    '''

    if len(a.shape) != 3 or len(b.shape) != 3:
        raise Exception(f"Expecting 3 dimensional inputs, got {a.shape} and {b.shape}")

    if a.shape != b.shape:
        raise Exception(f"Expecting inputs of the same shape, got {a.shape} and {b.shape}")

    tp = float(np.sum((b > 0)*(a > 0)))
    fn = float(np.sum((b > 0)*(a == 0)))

    if (tp + fn) == 0:
        return -1
    return tp / (tp + fn)