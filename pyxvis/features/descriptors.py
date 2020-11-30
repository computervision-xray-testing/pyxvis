from cv2 import xfeatures2d, FlannBasedMatcher
from numpy import array


def compute_sift(img):
    sift = xfeatures2d.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)

    keypoints = array([[k.pt[0], k.pt[1], k.size, k.angle] for k in keypoints])

    return keypoints, descriptors


def compute_brief(img):
    # Initiate FAST detector
    star = xfeatures2d.StarDetector_create()

    # Initiate BRIEF extractor
    brief = xfeatures2d.BriefDescriptorExtractor_create()

    # find the keypoints with STAR
    kp = star.detect(img, None)

    # compute the descriptors with BRIEF
    keypoints, descriptors = brief.compute(img, kp)
    keypoints = array([[k.pt[0], k.pt[1], k.size, k.angle] for k in keypoints])

    return keypoints, descriptors


descriptors_funcs = {
    'sift': compute_sift,
    'orb': [],
    'brief': compute_brief,
}


def compute_descriptors(input_img, descriptor_name, **opts):
    """
    A wrapper function of different methods to compute descriptors over an image. We develop this wrapper because the
    continuous changes in the interface of the methods and the restriction in some of the descriptors available in
    OpenCV, such as, SIFT and SURF. In this way, we expose the same interface for coding.

    Args:
        input_img (ndarray):
        descriptor_name (str:
        **opts: additional options

    Returns:
        kp (ndarray): Keypoints (N, 4), where N is the number of keypoints. Colums are x, y, scale and orientation.
        desc (ndarray): Descriptors (N, D), where N is the number of detected keypoints and D the size of the descriptor.

    [1] SIFT issues in OpenCV: https://stackoverflow.com/questions/52305578/sift-cv2-xfeatures2d-sift
    -create-not-working-even-though-have-contrib-install
    """
    # Descriptor names
    if descriptor_name not in descriptors_funcs.keys():
        raise ValueError('Invalid descriptor name')

    kp, desc = descriptors_funcs[descriptor_name](input_img)

    return kp, desc


def match_descriptors(descriptors1, descriptors2, matcher='flann', max_ratio=0.8):
    """

    Args:
        descriptors1:
        descriptors2:
        matcher:
        max_ratio:

    Returns:

    [1] https://stackoverflow.com/questions/30716610/how-to-get-pixel-coordinates-from-feature-matching-in-opencv-python
    """

    if matcher is 'flann':
        # FLANN parameters
        FLANN_INDEX_KDTREE = 0

        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # or pass empty dictionary

        flann = FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Need to draw only good matches, so create a mask using ratio test as per Lowe's paper.
    good_matches = []
    for i, (m, n) in enumerate(matches):
        if m.distance < max_ratio * n.distance:
            good_matches.append([n.queryIdx, m.trainIdx])  # Keep indexes of matched keypoints
    good_matches = array(good_matches)

    return good_matches
