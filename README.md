# blob-detector
An implementation of blob detector using LoG filter with python and opencv.
The code is enabling the user to choose the following parameters:

Blobs type- dark, bright, or both- paremeter 'type'.

Number of filters to apply- parameter 'levels'.
Initial sigma for first filter - paraemter 'sigma'.
The multiplicator - parameter 'k'.
The last 3 parameters determine the size of blobs that the agorithm will detect.

The threshold of the blobs which we want to detect according to - parameter 'threshold'.

The overlap threshold of 2 overlapping blobs, where we delete the smaller one between the two - parameter 'overlap_thresh'.

In the main part, the user can decide the all the parameter values, except the type.
