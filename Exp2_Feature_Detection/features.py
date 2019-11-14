import math

import cv2
import numpy as np
from scipy.ndimage import *
from scipy import ndimage, spatial

import transformations


def inbounds(shape, indices):
    assert len(shape) == len(indices)
    for i, ind in enumerate(indices):
        if ind < 0 or ind >= shape[i]:
            return False
    return True


## Keypoint detectors ##########################################################

class KeypointDetector(object):
    def detectKeypoints(self, image):
        """
        Input:
            image -- uint8 BGR image with values between [0, 255]
        Output:
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees), the detector response (Harris score for Harris detector)
            and set the size to 10.
        """
        raise NotImplementedError()


class DummyKeypointDetector(KeypointDetector):
    """
    Compute silly example features. This doesn't do anything meaningful, but
    may be useful to use as an example.
    """

    def detectKeypoints(self, image):
        """
        Input:
            image -- uint8 BGR image with values between [0, 255]
        Output:
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees), the detector response (Harris score for Harris detector)
            and set the size to 10.
        """
        image = image.astype(np.float32)
        image /= 255.
        features = []
        width, height = image.shape[:2]

        for y in range(height):
            for x in range(width):
                r = image[x, y, 0]
                g = image[x, y, 1]
                b = image[x, y, 2]

                if int(255 * (r + g + b) + 0.5) % 100 == 1:
                    # If the pixel satisfies this meaningless criterion,
                    # make it a feature.

                    f = cv2.KeyPoint()
                    f.pt = (x, y)
                    # Dummy size
                    f.size = 10
                    f.angle = 0
                    f.response = 10

                    features.append(f)

        return features


def gaussian_blur_kernel_2d(sigma, height, width):
    def gaussian_function(x, y, sigma_):
        pi = 3.1415926
        return (1 / (2 * pi * sigma_ ** 2)) * np.exp(-1 * (x ** 2 + y ** 2) / (2 * sigma_ ** 2))

    kernel = np.zeros(shape=(height, width))

    center_x = np.floor(width / 2)
    center_y = np.floor(height / 2)

    for i in range(width):
        for j in range(height):
            kernel[j, i] = gaussian_function(i - center_x, j - center_y, sigma)

    kernel = kernel / kernel.sum()

    return kernel


def cross_correlation_2d(img, kernel):
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)

    result_img = np.zeros(shape=img.shape)
    # 为后续处理方便

    img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REFLECT)

    for j in range(img.shape[1] - kernel.shape[1] + 1):
        for i in range(img.shape[0] - kernel.shape[0] + 1):
            result_img[i, j] = (img[i:(i + kernel.shape[0]), j:(j + kernel.shape[1])] * kernel).sum()
    result_img = np.squeeze(result_img)

    return result_img


def convolve_2d(img, kernel):
    kernel = np.flip(kernel)
    return cross_correlation_2d(img, kernel)


class HarrisKeypointDetector(KeypointDetector):

    def saveHarrisImage(self, harrisImage, srcImage):
        """
        Saves a visualization of the harrisImage, by overlaying the harris
        response image as red over the srcImage.

        Input:
            srcImage -- Grayscale input image in a numpy array with
                        values in [0, 1]. The dimensions are (rows, cols).
            harrisImage -- Grayscale input image in a numpy array with
                        values in [0, 1]. The dimensions are (rows, cols).
        """
        outshape = [harrisImage.shape[0], harrisImage.shape[1], 3]
        outImage = np.zeros(outshape)
        # Make a grayscale srcImage as a background
        srcNorm = srcImage * (0.3 * 255 / (np.max(srcImage) + 1e-50))
        outImage[:, :, :] = np.expand_dims(srcNorm, 2)

        # Add in the harris keypoints as red
        outImage[:, :, 2] += harrisImage * (4 * 255 / (np.max(harrisImage)) + 1e-50)
        cv2.imwrite("harris.png", outImage)

    # Compute harris values of an image.
    def computeHarrisValues(self, srcImage):
        """
        Input:
            srcImage -- Grayscale input image in a numpy array with
                        values in [0, 1]. The dimensions are (rows, cols).
        Output:
            harrisImage -- numpy array containing the Harris score at
                           each pixel.
            orientationImage -- numpy array containing the orientation of the
                                gradient at each pixel in degrees.
        """
        width, height = srcImage.shape[:2]

        harrisImage = np.zeros(srcImage.shape[:2], dtype='float')
        orientationImage = np.zeros(srcImage.shape[:2], dtype='float')

        # TODO 1: Compute the harris corner strength for 'srcImage' at
        # each pixel and store in 'harrisImage'.  See the project page
        # for direction on how to do this. Also compute an orientation
        # for each pixel and store it in 'orientationImage.'

        dx = sobel(srcImage, axis=1)
        dy = sobel(srcImage, axis=0)
        # dx = cv2.Sobel(srcImage, -1, dx=1, dy=0, ksize=3, borderType=cv2.BORDER_REFLECT)
        # dy = cv2.Sobel(srcImage, -1, dx=0, dy=1, ksize=3, borderType=cv2.BORDER_REFLECT)

        orientationImage = np.arctan2(dy, dx) * 180 / np.pi

        dxx = gaussian_filter(dx * dx, sigma=0.5)
        dyy = gaussian_filter(dy * dy, sigma=0.5)
        dxy = gaussian_filter(dx * dy, sigma=0.5)

        for i in range(0, width):
            for j in range(0, height):
                # print(i, j, width, height)
                a = dxx[i, j].astype(np.float32)
                b = dxy[i, j].astype(np.float32)
                c = dyy[i, j].astype(np.float32)

                # eigvalue, _ = np.linalg.eig(h)
                # if eigvalue.min() != 0:
                #     print(eigvalue.min())

                harrisImage[i, j] = (a * c - b * b) - 0.1 * np.square(a + c)
                # harrisImage[i, j] = eigvalue.min()
                # harrisImage[i, j] = eigvalue[0]*eigvalue[1]/(eigvalue[0]+eigvalue[1])
        # print(np.max(harrisImage), np.min(harrisImage))

        # #########删选7×7最大！！
        #
        # zuobiao = []
        #
        # harrisImage_ = np.zeros(harrisImage.shape, dtype='float')
        # harrisImage = np.pad(harrisImage, ((3, 3), (3, 3)), 'constant')
        #
        # for i in range(width):
        #     for j in range(height):
        #         if harrisImage[i + 3, j + 3] == np.max(harrisImage[i:i+7, j:j+7]):
        #             harrisImage_[i, j] = harrisImage[i + 3, j + 3]
        #             if harrisImage_[i, j] > 1e-3:
        #                 print(i, j, harrisImage_[i, j])
        #                 zuobiao.append((i, j))
        # harrisImage = np.copy(harrisImage_)
        # #
        # # ###### 调试方向
        # #
        # print(len(zuobiao))
        #
        # for a in zuobiao:
        #     print(a, orientationImage[a]*180/math.pi)
        #     cv2.circle(harrisImage, a, 5, (255, 255, 255), 8)
        #     d = 30
        #     print((int(a[0]), int(a[1])), (int(a[0]+d*np.cos(orientationImage[a])), int(a[1]+d*np.sin(orientationImage[a]))))
        #     cv2.line(harrisImage, (int(a[0]), int(a[1])), (int(a[0]+d*np.cos(orientationImage[a])), int(a[1]+d*np.sin(orientationImage[a]))), (255, 255, 255))
        #     # cv2.dra
        #     cv2.imshow("iner", harrisImage * 255)
        #     transMx = cv2.getRotationMatrix2D(a, orientationImage[a]*180/math.pi, 1)
        #     destx, desty = tuple([i for i in np.dot(transMx, np.array([[a[0]], [a[1]], [1]])).reshape(2)])
        #
        #     destx, desty = int(destx), int(desty)
        #
        #     destImage = cv2.warpAffine(harrisImage, transMx, (harrisImage.shape[1], harrisImage.shape[0]),
        #                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        #
        #     cv2.circle(destImage, (destx, desty), 20, (255, 255, 255), 2)
        #
        # #     destArae = np.copy(destImage[destx - 20:destx + 20, desty - 20:desty + 20])
        # #     print("destArae.shape1", destArae.shape)
        #
        #     # aaa=cv2.GaussianBlur(destArae, (33, 33), 10)
        #     # print("aaa.shape2", aaa.shape, aaa.max())
        #     # cv2.imshow("aaa", aaa * 255)
        #
        #     # print(len(destArae))
        #
        #
        #     cv2.imshow("iner_rot", destImage * 255)
        #     harrisImage = np.copy(harrisImage_)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()

        # print(harrisImage.shape, srcImage.shape)

        # Save the harris image as harris.png for the website assignment
        self.saveHarrisImage(harrisImage, srcImage)

        return harrisImage, orientationImage

    def computeLocalMaxima(self, harrisImage):
        """
        Input:
            harrisImage -- numpy array containing the Harris score at
                           each pixel.
        Output:
            destImage -- numpy array containing True/False at
                         each pixel, depending on whether
                         the pixel value is the local maxima in
                         its 7x7 neighborhood.
        """
        destImage = np.zeros_like(harrisImage, np.bool)

        # TODO 2: Compute the local maxima image

        # destImage = np.zeros(shape=harrisImage.shape)

        width, height = harrisImage.shape[:2]
        harrisImage_ = np.pad(harrisImage, ((3, 3), (3, 3)), 'constant')

        for i in range(0, width):
            for j in range(0, height):
                if harrisImage_[i + 3, j + 3] == np.max(harrisImage_[i:i + 7, j:j + 7]):
                    destImage[i, j] = True

        return destImage

    def detectKeypoints(self, image):
        """
        Input:
            image -- BGR image with values between [0, 255]
        Output:
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees), the detector response (Harris score for Harris detector)
            and set the size to 10.
        """
        image = image.astype(np.float32)
        image /= 255.
        width, height = image.shape[:2]
        features = []

        # Create grayscale image used for Harris detection
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # computeHarrisValues() computes the harris score at each pixel
        # position, storing the result in harrisImage.
        # You will need to implement this function.
        harrisImage, orientationImage = self.computeHarrisValues(grayImage)

        # Compute local maxima in the Harris image.  You will need to
        # implement this function. Create image to store local maximum harris
        # values as True, other pixels False
        harrisMaxImage = self.computeLocalMaxima(harrisImage)

        # Loop through feature points in harrisMaxImage and fill in information
        # needed for descriptor computation for each point.
        # You need to fill x, y, and angle.
        # print(orientationImage.shape, height, width)
        for x in range(width):
            for y in range(height):
                if not harrisMaxImage[x, y]:
                    continue
                f = cv2.KeyPoint()
                # TODO 3: Fill in feature f with location and orientation
                # data here. Set f.size to 10, f.pt to the (x,y) coordinate,
                # f.angle to the orientation in degrees and f.response to
                # the Harris score
                f.size = 10
                f.pt = (y, x)
                f.angle = orientationImage[x, y]
                f.response = harrisImage[x, y]

                features.append(f)

        # print(len(features))
        def distance(pt1, pt2):
            return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)

        l = len(features)
        local_max = []
        raduis = np.array([99999] * l)
        # print(raduis)
        max_res = 0
        for f in features:
            if f.response > max_res:
                max_res = f.response
            local_max.append(f.response * 0.9)

        max_res *= 0.9
        for i in range(l):
            res = features[i].response
            # r =
            # p = cv2.KeyPoint()
            # print(i)
            if res > max_res:
                raduis[i] = 99999
            else:
                for j in range(l):
                    if local_max[j] > res:
                        # print(i, j)
                        d = distance(features[i].pt, features[j].pt)
                        if raduis[i] > d:
                            raduis[i] = d

        n_ip = 1750
        r_sort = np.argsort(raduis)
        rpts = []
        for i in range(l):
            rpts.append(features[r_sort[i]])
        print("共有特征点", len(rpts))

        return rpts[::-1][:n_ip]


class ORBKeypointDetector(KeypointDetector):
    def detectKeypoints(self, image):
        '''
        Input:
            image -- uint8 BGR image with values between [0, 255]
        Output:
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees) and set the size to 10.
        '''
        detector = cv2.ORB_create()
        return detector.detect(image, None)


## Feature descriptors #########################################################


class FeatureDescriptor(object):
    # Implement in child classes
    def describeFeatures(self, image, keypoints):
        """
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
        Output:
            Descriptor numpy array, dimensions:
                keypoint number x feature descriptor dimension
        """
        raise NotImplementedError


class SimpleFeatureDescriptor(FeatureDescriptor):
    # TODO: Implement parts of this function
    def describeFeatures(self, image, keypoints):
        """
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
                         descriptors at the specified coordinates
        Output:
            desc -- K x 25 numpy array, where K is the number of keypoints
        """
        image = image.astype(np.float32)
        image /= 255.
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        desc = np.zeros((len(keypoints), 5 * 5))

        height, width = image.shape[:2]

        pad = int((5 - 1) / 2)
        grayImage_ = np.pad(grayImage, ((pad, pad), (pad, pad)), 'constant')

        for i, f in enumerate(keypoints):
            y, x = f.pt
            x, y = int(np.round(x)), int(np.round(y))

            # TODO 4: The simple descriptor is a 5x5 window of intensities
            # sampled centered on the feature point. Store the descriptor
            # as a row-major vector. Treat pixels outside the image as zero.
            try:
                desc[i] = grayImage_[x: x + 5, y:y + 5].reshape(25)
            except ValueError:
                print(grayImage_.shape, x, y)
                print("跳过一个错误")
                desc[i] = desc[i] = np.array([0 for _ in range(25)])

        return desc


class MOPSFeatureDescriptor(FeatureDescriptor):
    # TODO: Implement parts of this function
    def describeFeatures(self, image, keypoints):
        """
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
        Output:
            desc -- K x W^2 numpy array, where K is the number of keypoints
                    and W is the window size
        """
        image = image.astype(np.float32)
        image /= 255.
        # This image represents the window around the feature you need to
        # compute to store as the feature descriptor (row-major)
        windowSize = 8
        desc = np.zeros((len(keypoints), windowSize * windowSize))
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grayImage = ndimage.gaussian_filter(grayImage, 0.5)

        for i, f in enumerate(keypoints):
            # TODO 5: Compute the transform as described by the feature
            # location/orientation. You will need to compute the transform
            # from each pixel in the 40x40 rotated window surrounding
            # the feature to the appropriate pixels in the 8x8 feature
            # descriptor image.

            t1 = np.array([[1., 0., -1. * f.pt[0]], [0., 1., -1. * f.pt[1]], [0., 0., 1.]])
            r = np.array([[np.cos(f.angle * np.pi / 180.), np.sin(f.angle * np.pi / 180.), 0.],
                          [-1 * np.sin(f.angle * np.pi / 180.), np.cos(f.angle * np.pi / 180.), 0.],
                          [0., 0., 1.]])
            # s = np.array([[1. / 5., 0., 0.], [0., 1. / 5., 0.], [0., 0., 1.]])
            # t2 = np.array([[1., 0., 4.], [0., 1., 4.], [0., 0., 1.]])
            t2 = np.array([[1., 0., 32.], [0., 1., 32.], [0., 0., 1.]])
            # transMx = t2.dot(s).dot(r).dot(t1)[:2, :]
            transMx = t2.dot(r).dot(t1)[:2, :]

            destImage = cv2.warpAffine(grayImage, transMx, (64, 64),
                                       flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

            destImage = cv2.pyrDown(destImage)
            destImage = cv2.pyrDown(destImage)
            destImage = cv2.pyrDown(destImage)

            # TODO 6: Normalize the descriptor to have zero mean and unit
            # variance. If the variance is zero then set the descriptor
            # # vector to zero. Lastly, write the vector to desc.

            destArae = destImage.flatten()

            sig = np.std(destArae)

            if sig <= 1e-5:
                desc[i] = np.zeros(64)
            else:
                desc[i] = ((destArae - np.mean(destArae)) / sig)

        return desc


class ORBFeatureDescriptor(KeypointDetector):
    def describeFeatures(self, image, keypoints):
        """
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
        Output:
            Descriptor numpy array, dimensions:
                keypoint number x feature descriptor dimension
        """
        descriptor = cv2.ORB_create()
        kps, desc = descriptor.compute(image, keypoints)
        if desc is None:
            desc = np.zeros((0, 128))

        return desc


# Compute Custom descriptors (extra credit)
class CustomFeatureDescriptor(FeatureDescriptor):
    def describeFeatures(self, image, keypoints):
        """
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
        Output:
            Descriptor numpy array, dimensions:
                keypoint number x feature descriptor dimension
        """
        raise NotImplementedError('NOT IMPLEMENTED')


## Feature matchers ############################################################


class FeatureMatcher(object):
    def matchFeatures(self, desc1, desc2):
        '''
        Input:
            desc1 -- the feature descriptors of image 1 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
            desc2 -- the feature descriptors of image 2 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
        Output:
            features matches: a list of cv2.DMatch objects
                How to set attributes:
                    queryIdx: The index of the feature in the first image
                    trainIdx: The index of the feature in the second image
                    distance: The distance between the two features
        '''
        raise NotImplementedError

    # Evaluate a match using a ground truth homography.  This computes the
    # average SSD distance between the matched feature points and
    # the actual transformed positions.
    @staticmethod
    def evaluateMatch(features1, features2, matches, h):
        d = 0
        n = 0

        for m in matches:
            id1 = m.queryIdx
            id2 = m.trainIdx
            ptOld = np.array(features2[id2].pt)
            ptNew = FeatureMatcher.applyHomography(features1[id1].pt, h)

            # Euclidean distance
            d += np.linalg.norm(ptNew - ptOld)
            n += 1

        return d / n if n != 0 else 0

    # Transform point by homography.
    @staticmethod
    def applyHomography(pt, h):
        x, y = pt
        d = h[6] * x + h[7] * y + h[8]

        return np.array([(h[0] * x + h[1] * y + h[2]) / d,
                         (h[3] * x + h[4] * y + h[5]) / d])


class SSDFeatureMatcher(FeatureMatcher):
    def matchFeatures(self, desc1, desc2):
        '''
        Input:
            desc1 -- the feature descriptors of image 1 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
            desc2 -- the feature descriptors of image 2 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
        Output:
            features matches: a list of cv2.DMatch objects
                How to set attributes:
                    queryIdx: The index of the feature in the first image
                    trainIdx: The index of the feature in the second image
                    distance: The distance between the two features
        '''
        matches = []
        # feature count = n
        assert desc1.ndim == 2
        # feature count = m
        assert desc2.ndim == 2
        # the two features should have the type
        assert desc1.shape[1] == desc2.shape[1]

        if desc1.shape[0] == 0 or desc2.shape[0] == 0:
            return []

        # TODO 7: Perform simple feature matching.  This uses the SSD
        # distance between two feature vectors, and matches a feature in
        # the first image with the closest feature in the second image.
        # Note: multiple features from the first image may match the same
        # feature in the second image.
        for i, f1 in enumerate(desc1):
            matche = cv2.DMatch()
            matche.queryIdx = i
            for j, f2 in enumerate(desc2):
                dist = np.sum(np.square(f1 - f2))
                if dist < matche.distance:
                    matche.distance = dist
                    matche.trainIdx = j
            matches.append(matche)

        return matches


class RatioFeatureMatcher(FeatureMatcher):
    def matchFeatures(self, desc1, desc2):
        """
        Input:
            desc1 -- the feature descriptors of image 1 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
            desc2 -- the feature descriptors of image 2 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
        Output:
            features matches: a list of cv2.DMatch objects
                How to set attributes:
                    queryIdx: The index of the feature in the first image
                    trainIdx: The index of the feature in the second image
                    distance: The ratio test score
        """
        matches = []
        # feature count = n
        assert desc1.ndim == 2
        # feature count = m
        assert desc2.ndim == 2
        # the two features should have the type
        assert desc1.shape[1] == desc2.shape[1]

        if desc1.shape[0] == 0 or desc2.shape[0] == 0:
            return []

        # TODO 8: Perform ratio feature matching.
        # This uses the ratio of the SSD distance of the two best matches
        # and matches a feature in the first image with the closest feature in the
        # second image.
        # Note: multiple features from the first image may match the same
        # feature in the second image.
        # You don't need to threshold matches in this function

        for i, f1 in enumerate(desc1):
            # matche = cv2.DMatch()
            # matche.queryIdx = i
            distance_list = [(j, np.sum(np.square(f1 - f2))) for j, f2 in enumerate(desc2)]

            distance_list.sort(key=lambda x: x[1])

            # try:
            d = distance_list[0][1] / distance_list[1][1]
            # except SystemError:
            #     pass
            # matche.trainIdx = distance_list[0][0]

            matche = cv2.DMatch(_queryIdx=i, _trainIdx=distance_list[0][0], _distance=d)
            matches.append(matche)

        return matches


class ORBFeatureMatcher(FeatureMatcher):
    def __init__(self):
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        super(ORBFeatureMatcher, self).__init__()

    def matchFeatures(self, desc1, desc2):
        return self.bf.match(desc1.astype(np.uint8), desc2.astype(np.uint8))


if __name__ == '__main__':

    img1 = cv2.imread("testbw2.png")
    img2 = cv2.imread("./resources/triangle2.jpg")

    image = img2.astype(np.float32)
    # image /= 255.

    # Create grayscale image used for Harris detection
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hkpd = HarrisKeypointDetector()
    a = MOPSFeatureDescriptor()
    fs = hkpd.detectKeypoints(image)

    for f in fs:
        if abs(f.response) < 1e-5:
            continue

        t1 = np.array([[1., 0., -1. * f.pt[1]], [0., 1., -1. * f.pt[0]], [0., 0., 1.]])
        print(f.angle)
        # r = np.vstack((cv2.getRotationMatrix2D((0, 0), f.angle, 1), np.array([0., 0., 1.])))
        # print("r\n", r)
        r = np.array([[np.cos(f.angle * np.pi / 180.), np.sin(f.angle * np.pi / 180.), 0.],
                      [-1 * np.sin(f.angle * np.pi / 180.), np.cos(f.angle * np.pi / 180.), 0.],
                      [0., 0., 1.]])

        t2 = np.array([[1., 0., 32.], [0., 1., 32.], [0., 0., 1.]])
        # s = np.array([[1, 0., 0.], [0., 1, 0.], [0., 0., 1.]])

        transMx = t2.dot(r).dot(t1)[:2, :]

        d = 30

        cv2.circle(img2, (int(f.pt[1]), int(f.pt[0])), 20, (255, 255, 255), 2)

        cv2.line(img2, (int(f.pt[1]), int(f.pt[0])),
                 (int(f.pt[1] + d * np.cos(f.angle / 180 * np.pi)), int(f.pt[0] + d * np.sin(f.angle / 180 * np.pi))),
                 (255, 255, 255), 10)

        m = transMx.dot(np.array([[f.pt[1]], [f.pt[0]], [1]]))
        destx = m[0][0]
        desty = m[1][0]
        destImage = cv2.warpAffine(img2, transMx, (64, 64),
                                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        dst = cv2.pyrDown(destImage)
        dst = cv2.pyrDown(dst)
        dst = cv2.pyrDown(dst)
        print(dst.shape)

        cv2.imshow('dst', dst)

        cv2.circle(destImage, (int(destx), int(desty)), 20, (255, 255, 255), 2)

        #
        destImage2 = cv2.warpAffine(grayImage, transMx, (40, 40),
                                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        sig = np.std(destImage2)
        #
        destArae = ((destImage2 - np.mean(destImage2)) / sig)

        cv2.imshow('img2', destImage)
        cv2.imshow('img', img2)
        cv2.imshow('img3', destArae)
        print(destArae)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
