import cv2
import math
import numpy as np
from matplotlib import pyplot as plt
from LoG_filter import log_filt
from scipy import spatial

# The parameters we will use to apply the blobs detector on the image
sigma = 1
k = 1.414
number_of_levels = 12
overlap_thresh = 0.5
threshold = 0.21

def display_image_and_wait(image, title):
    """
        displays an image and wait for any key.
        Parameters
        ----------
        image : the image to display
        title : the title of the image

        Returns
        -------
        None
        """
    # display the image and wait for any key
    cv2.imshow(title, image)
    cv2.waitKey()
    cv2.destroyAllWindows()


def blob_overlap(blob1, blob2):
    """
    calculates the percentage of overlapping of 2 given blobs.
    Parameters
    ----------
    blob1 : vector of length 3
        first blob.
    blob2 : vector of length 3
        second blob.

    Returns
    -------
    fraction: float
        Fraction of area of the overlap between the two circle.
    """

    # calculating the radius of each blob
    r1 = blob1[2]
    r2 = blob2[2]

    # calculating the distance between the 2 centers
    d = math.sqrt(np.sum((blob1[:-1] - blob2[:-1]) ** 2))

    # there's no overlap between the 2 given blobs
    if d > r1 + r2:
        return 0
    # one blob is inside the other. we return 1 because it's a full overlap.
    elif d <= abs(r1 - r2):
        return 1
    # partial overlap. we calculate the area of overlapping
    # the calculation is based on https://github.com/scikit-image/scikit-image/blob/master/skimage/feature/blob.py#L335
    # the function _compute_disk_overlap
    else:
        ratio1 = (d ** 2 + r1 ** 2 - r2 ** 2) / (2 * d * r1)
        ratio1 = np.clip(ratio1, -1, 1)
        acos1 = math.acos(ratio1)

        ratio2 = (d ** 2 + r2 ** 2 - r1 ** 2) / (2 * d * r2)
        ratio2 = np.clip(ratio2, -1, 1)
        acos2 = math.acos(ratio2)

        a = -d + r2 + r1
        b = d - r2 + r1
        c = d + r2 - r1
        d = d + r2 + r1

        area = (r1 ** 2 * acos1 + r2 ** 2 * acos2 - 0.5 * math.sqrt(abs(a * b * c * d)))
        return area / (math.pi * (min(r1, r2) ** 2))


def redundancy(blobs_array, overlap):
    """
        omitting overlapping blobs.
        Parameters
        ----------
        blobs_array : an array of shape (n, 3) where n is the number of blobs.
        overlap : overlap threshold to omit blobs according to.

        Returns
        -------
        array of blobs after omitting overlapped blobs.
        """
    # creating a list of pairs of blobs for comparison.
    sigma = blobs_array[:, -1].max()
    distance = 2 * sigma * math.sqrt(blobs_array.shape[1] - 1)
    tree = spatial.cKDTree(blobs_array[:, :-1])
    pairs = np.array(list(tree.query_pairs(distance)))

    # if there are no pairs
    if len(pairs) == 0:
        return blobs_array

    # if there are some pairs, we omit overlapped
    else:
        for (i, j) in pairs:
            blob1, blob2 = blobs_array[i], blobs_array[j]
            if blob_overlap(blob1, blob2) > overlap:
                # deleting the smaller detected blob
                if blob1[-1] > blob2[-1]:
                    blob2[-1] = 0
                else:
                    blob1[-1] = 0

    return np.array([b for b in blobs_array if b[-1] > 0])


def detect_blobs(multi_LoG_matrix, thresh):
    """
        detecting blobs in an image, given its Laplacian of Gaussian matrix, and a threshold value
        for the intensity.
        ----------
        multi_LoG_matrix : The Laplacian of Gaussian matrix to find blobs in.
        thresh : the threshold value to decide if save the blobs or ignore it.

        Returns
        -------
        blobs: list of size (n, 3)
            A list of detected blobs.
        """
    blobs = list()
    rows = multi_LoG_matrix.shape[0]
    cols = multi_LoG_matrix.shape[1]
    # going through all the 3X3Xn neighbourhoods
    for i in range(1, rows):
        for j in range(1, cols):
            # slicing a neighbourhood
            slice_img = multi_LoG_matrix[i - 1:i + 2, j - 1:j + 2, :]
            # save the max value of the neighbourhood
            result = np.amax(slice_img)
            # if the max is greater than the threshold and it is in the middle of the neighbourhood
            # we keep it as a blobs, with a radius proportional to its level
            if result > thresh:
                x, y, z = np.unravel_index(slice_img.argmax(), slice_img.shape)
                if y == 1 and x == 1:
                    blobs.append((i + x - 1, j + y - 1, int(round(k ** (z+1) * sigma))))
    return blobs


def create_filter_set(initial_sigma=1, k_const=1.414, levels=10):
    """
        create a set of filters with the given parametes
        Parameters
        ----------
        initial_sigma : the sigma value of the first LoG filter.
        k_const : the value in which we multiply the previous sigma to get
            the next filter's sigma.
        levels : the total number of filters to create.
        Returns
        -------
        fraction: list
            The filters set.
        """
    # creating the set of filters to apply on the image
    current_sigma = initial_sigma / k_const
    filter_set = list()
    sigma_array = list()

    for i in range(0, levels):
        # calculate current filter's sigma
        current_sigma = current_sigma * k
        sigma_array.append(current_sigma)

        # calculate current filter size
        filt_size = 2 * np.ceil(3 * current_sigma) + 1  # filter size

        # creating current filter
        h = log_filt(filt_size, current_sigma) * (current_sigma ** 2)
        filter_set.append(h)

    return filter_set, sigma_array


def present_filter_set(levels, sigma_array, filter_set):
    """
        Presents the filters set.
        Parameters
        ----------
        levels : the amount of filters in the set.
        sigma_array : an array of the sigma values of the filter.

        Returns
        -------
        None
        """
    # displaying all the different filters we will aplly on the image
    fig_filt = plt.figure(figsize=(20, 20))
    rows = int(math.floor(levels / 5))
    columns = int(math.ceil(levels / rows))

    # going through the filters and add them to the plot
    for i in range(1, levels + 1):
        filt = filter_set[i - 1]
        fig_filt.add_subplot(rows, columns, i).title.set_text('sigma:' + str(round(sigma_array[i - 1], 3)))
        plt.imshow(filt, cmap='gray')
    plt.show()


def present_convolved_images(multi_LoG_matrix, levels, sigma_array):
    """
        Presents the convolved images.
        Parameters
        ----------
        multi_LoG_matrix : the convolved images matrix.
        levels : the number of convolved images.
        sigma_array : an array with the sigma values of the convolved images.

        Returns
        -------
        None
        """
    # displaying all the results of applying the different filters on the image
    fig = plt.figure(figsize=(20, 20))
    rows = int(math.floor(levels / 5))
    columns = int(math.ceil(levels / rows))

    for i in range(1, levels + 1):
        img = multi_LoG_matrix[:, :, i - 1]
        fig.add_subplot(rows, columns, i).title.set_text('sigma:' + str(round(sigma_array[i - 1], 3)))
        plt.imshow(img, cmap='gray')
    plt.show()


def create_multi_LoG_matrix(gray_image, blob_type, filter_set):
    """
        calculates convolved image for each filter, and stores it all in a matrix.
        Parameters
        ----------
        gray_image : the image to convolve with the filters.
        blob_type : the type of blobs we want to detect.
        filter_set : the set of LoG filters to apply on the image.

        Returns
        -------
        multi_LoG_matrix: the matrix that contains the convolved images.
        """
    # creating the matrix to hold all the convolution results
    levels = len(filter_set)
    multi_LoG_matrix = np.zeros((gray_image.shape[0], gray_image.shape[1], levels))

    # choose which kind of blobs to discover.
    if blob_type == 'both':
        for i in range(0, len(filter_set)):
            multi_LoG_matrix[:, :, i] = abs(cv2.filter2D(gray_image, -1, filter_set[i], borderType=cv2.BORDER_REFLECT))
    else:
        if blob_type == 'dark':
            const = 1
        elif blob_type == 'bright':
            const = -1
        else:
            print('There is no such type of blobs. please select dark, bright or both.')
            raise SystemExit
        for i in range(0, len(filter_set)):
            multi_LoG_matrix[:, :, i] = const * cv2.filter2D(gray_image, -1, filter_set[i], borderType=cv2.BORDER_REFLECT)

    return multi_LoG_matrix


def blob_detector(image, initial_sigma=1, k_const=1.414, levels=10, overlap_threshold=0.5, blobs_threshold=0.21,
                  blob_type='both'):
    """
        finds blobs in a given image.
        ----------
        image : the image to find blobs in. could be of any number of channels.
        initial_sigma : the sigma of the first and smallest LoG filter.
        k_const : the constant which we multiply the previous sigma value
            to create the new sigma value.
        levels : the number of filters we want to apply on the image.
        overlap_threshold : the overlapped area of 2 blobs, where we consider the 2 overlapping.
        blobs_threshold : the minimum intensity for a pixel to consider a blobs.
        blob_type : the type of blobs we want to find. could be 'dark', 'bright', or 'both'.

        Returns
        -------
        reduced_blobs_list
            A list of blobs found in the image, according to all chosen parameters.
            each blob has a center(x,y) and a radius.
        """
    # converting the image to grayscale and scaling it
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 255.0
    display_image_and_wait(gray_image, title='grayscale')

    filter_set, sigma_array = create_filter_set(initial_sigma=initial_sigma, k_const=k_const, levels=levels)

    # uncomment this part, to view the different filters that will be applied on the image.
    #present_filter_set(levels=levels, sigma_array=sigma_array, filter_set=filter_set)

    # calculating the convolved images with the filter. meaning applying the filter on the image
    multi_LoG_matrix = create_multi_LoG_matrix(gray_image=gray_image, blob_type=blob_type, filter_set=filter_set)

    # uncomment this part, to view how the different filters are applied on the image.
    #present_convolved_images(multi_LoG_matrix=multi_LoG_matrix, levels=levels, sigma_array=sigma_array)

    # finding the blobs in the image and print the result
    blobs = list()
    blobs = detect_blobs(multi_LoG_matrix, blobs_threshold)
    print('amount of blobs detected: ' + str(len(blobs)))
    blobs_set = list(set(blobs))
    blobs_set = np.array(blobs_set)

    # reducing overlapped blobs and print the final result
    reduced_blobs_list = redundancy(blobs_set, overlap_threshold)
    print('amount of blobs left after reduction of overlapped blobs: ' + str(len(reduced_blobs_list)))

    # paint circles on the original image, according to the blobs found
    final_image = image
    for i in range(0, len(reduced_blobs_list)):
        center = reduced_blobs_list[i][1], reduced_blobs_list[i][0]
        radius = reduced_blobs_list[i][2]
        final_image = cv2.circle(final_image, center, radius, color=(0, 0, 255), thickness=1, lineType=8, shift=0)

    # print the final image with the detected blobs
    display_image_and_wait(final_image, title='image with detected blobs')

    return reduced_blobs_list


# the main program will detect blobs in the list of images and demonstrate them.
names = {0: 'sunflowers.jpg', 1: 'fishes.jpg', 2: 'einstein.jpg', 3: 'butterfly.jpg', 4: 'blobs.jpg', 5: 'cool.jpg'}
number_of_images = len(names)

# going through the images and apply the algorithm on each one of them.
for ind in range(0, number_of_images):
    img = cv2.imread(names[ind])
    display_image_and_wait(img, title='original')
    blob_detector(img, initial_sigma=sigma, k_const=k, levels=number_of_levels, overlap_threshold=overlap_thresh,
                  blobs_threshold=threshold, blob_type='both')



