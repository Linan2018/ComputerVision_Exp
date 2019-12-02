import math
import sys

import cv2
import numpy as np


class ImageInfo:
    def __init__(self, name, img, position):
        self.name = name
        self.img = img
        self.position = position


def imageBoundingBox(img, M):
    """
       This is a useful helper function that you might choose to implement
       that takes an image, and a transform, and computes the bounding box
       of the transformed image.

       INPUT:
         img: image to get the bounding box of
         M: the transformation to apply to the img
       OUTPUT:
         minX: int for the minimum X value of a corner
         minY: int for the minimum Y value of a corner
         maxX: int for the maximum X value of a corner
         maxY: int for the maximum Y value of a corner
    """
    # TODO 8
    # TODO-BLOCK-BEGIN
    # print("start TODO 8  imageBoundingBox")

    h = img.shape[0] - 1
    w = img.shape[1] - 1

    left_top = np.array([[0], [0], [1]])
    right_top = np.array([[w], [0], [1]])
    left_bottom = np.array([[0], [h], [1]])
    right_bottom = np.array([[w], [h], [1]])

    left_top, right_top, left_bottom, right_bottom = \
        tuple(map(lambda v: M.dot(v) / M.dot(v)[-1], [left_top, right_top, left_bottom, right_bottom]))

    minX = min(left_top[0], right_top[0], left_bottom[0], right_bottom[0])
    minY = min(left_top[1], right_top[1], left_bottom[1], right_bottom[1])
    maxX = max(left_top[0], right_top[0], left_bottom[0], right_bottom[0])
    maxY = max(left_top[1], right_top[1], left_bottom[1], right_bottom[1])

    # TODO-BLOCK-END
    return int(minX), int(minY), int(maxX), int(maxY)


def accumulateBlend(img, acc, M, blendWidth):
    """
       INPUT:
         img: image to add to the accumulator
         acc: portion of the accumulated image where img should be added
         M: the transformation mapping the input image to the accumulator
         blendWidth: width of blending function. horizontal hat function
       OUTPUT:
         modify acc with weighted copy of img added where the first
         three channels of acc record the weighted sum of the pixel colors
         and the fourth channel of acc records a sum of the weights
    """
    # BEGIN TODO 10
    # Fill in this routine
    # TODO-BLOCK-BEGIN
    # print("start TODO 10 accumulateBlend   add  acc")
    # print("img.shape:", img.shape)
    # print("acc.shape:", acc.shape)
    minX, minY, maxX, maxY = imageBoundingBox(img, M)
    # print("M:\n", M)
    # print("img.shape", img.shape)
    width, height = maxX - minX, maxY - minY
    # print("height, width", height, width)

    x_mesh, y_mesh = np.meshgrid(range(minX, maxX), range(minY, maxY))

    grids = np.array([[*pt, 1] for pt in zip(x_mesh.flatten(), y_mesh.flatten())]).T
    # print("grids.shape", grids.shape)
    pre_grids = np.dot(np.linalg.inv(M), grids)  # 反卷绕

    pre_grids /= pre_grids[-1]
    # print("pre_grids.shape", pre_grids.shape)

    src = cv2.remap(img, pre_grids[0].reshape(height, width).astype(np.float32),
                    pre_grids[1].reshape(height, width).astype(np.float32), cv2.INTER_LINEAR)
    # print("src.shape:", src.shape)
    # 权重w

    # cv2.imwrite("img.png", src)
    # cv2.imshow("src", src[:, :, :3].astype(np.uint8))
    # cv2.waitKey(0)  # 暂停窗口

    w = np.ones(shape=(height, width))

    blend = np.linspace(1, 1 / blendWidth, blendWidth)

    w[:, :blendWidth] = np.flip(blend)
    w[:, -1 * blendWidth:] = blend

    dst = np.multiply(src, np.expand_dims(w, -1))

    # cv2.imshow("w", np.expand_dims(w, -1).astype(np.uint8))
    # cv2.waitKey(0)  # 暂停窗口

    # print("dst.shape:", dst.shape)
    # print("minX: maxX+1", maxX+1 - minX)
    # print("acc[minY: maxY, minX: maxX, :3].shape:", acc[minY: maxY, minX: maxX, :3].shape)
    # print("minY, maxY, minX, maxX", minY, maxY, minX, maxX)

    # cv2.imwrite("img.png", img)
    # cv2.imwrite("dst.png", dst[:, :, :3].astype(np.uint8))
    # cv2.imshow("dst", dst[:, :, :3].astype(np.uint8))
    # cv2.imshow("img", img)
    # cv2.waitKey(0)  # 暂停窗口

    acc[minY: maxY, minX: maxX, :3] += dst
    acc[minY: maxY, minX: maxX, 3] += np.multiply(w, np.sum(dst, axis=2) > 0)
    # acc[minY: maxY, minX: maxX, 3] += w

    # cv2.imwrite("acc.png", acc[:, :, :3].astype(np.uint8))
    # cv2.imshow("acc", acc[:, :, :3].astype(np.uint8))
    # cv2.waitKey(0)  # 暂停窗口

    # TODO-BLOCK-END
    # END TODO

    return acc


def normalizeBlend(acc):
    """
       INPUT:
         acc: input image whose alpha channel (4th channel) contains
         normalizing weight values
       OUTPUT:
         img: image with r,g,b values of acc normalized
    """
    # BEGIN TODO 11
    # fill in this routine..
    # TODO-BLOCK-BEGIN
    # print("start TODO 11  normalizeBlend")

    height, width = acc.shape[:2]

    img = np.zeros(shape=(height, width, 3))

    for i in range(height):
        for j in range(width):
            if acc[i, j, 3] > 0:
                img[i, j] = acc[i, j, :3] / acc[i, j, 3]
            else:
                img[i, j] = np.array([0, 0, 0])

    img = img.astype(np.uint8)

    # TODO-BLOCK-END
    # END TODO
    return img


def getAccSize(ipv):
    """
       This function takes a list of ImageInfo objects consisting of images and
       corresponding transforms and Returns useful information about the accumulated
       image.

       INPUT:
         ipv: list of ImageInfo objects consisting of image (ImageInfo.img) and transform(image (ImageInfo.position))
       OUTPUT:
         accWidth: Width of accumulator image(minimum width such that all tranformed images lie within acc)
         accHeight: Height of accumulator image(minimum height such that all tranformed images lie within acc)

         channels: Number of channels in the accumulator image
         width: Width of each image(assumption: all input images have same width)
         translation: transformation matrix so that top-left corner of accumulator image is origin
    """

    # Compute bounding box for the mosaic
    minX = sys.maxsize
    minY = sys.maxsize
    maxX = 0
    maxY = 0
    channels = -1
    width = -1  # Assumes all images are the same width
    M = np.identity(3)
    # print("start TODO 9 getAccSize", "including", len(ipv), "images")
    for i in ipv:
        M = i.position
        img = i.img
        _, w, c = img.shape
        if channels == -1:
            channels = c
            width = w

        # BEGIN TODO 9
        # add some code here to update minX, ..., maxY
        # TODO-BLOCK-BEGIN

        x1, y1, x2, y2 = imageBoundingBox(img, M)
        minX = np.minimum(x1, minX)
        minY = np.minimum(y1, minY)
        maxX = np.maximum(x2, maxX)
        maxY = np.maximum(y2, maxY)
        # TODO-BLOCK-END
        # END TODO

    # Create an accumulator image
    accWidth = int(math.ceil(maxX) - math.floor(minX))
    accHeight = int(math.ceil(maxY) - math.floor(minY))
    # print('accWidth, accHeight:', (accWidth, accHeight))
    translation = np.array([[1, 0, -minX], [0, 1, -minY], [0, 0, 1]])

    return accWidth, accHeight, channels, width, translation


def pasteImages(ipv, translation, blendWidth, accWidth, accHeight, channels):
    print(accHeight * accWidth * (channels + 1))
    acc = np.zeros((accHeight, accWidth, channels + 1))
    # Add in all the images
    M = np.identity(3)
    for count, i in enumerate(ipv):
        M = i.position
        img = i.img

        M_trans = translation.dot(M)
        accumulateBlend(img, acc, M_trans, blendWidth)

    return acc


def getDriftParams(ipv, translation, width):
    # Add in all the images
    M = np.identity(3)
    x_init, y_init, x_final, y_final = 0, 0, 0, 0
    for count, i in enumerate(ipv):
        if count != 0 and count != (len(ipv) - 1):
            continue

        M = i.position

        M_trans = translation.dot(M)

        p = np.array([0.5 * width, 0, 1])
        p = M_trans.dot(p)

        # First image
        if count == 0:
            x_init, y_init = p[:2] / p[2]
        # Last image
        if count == (len(ipv) - 1):
            x_final, y_final = p[:2] / p[2]

    return x_init, y_init, x_final, y_final


def computeDrift(x_init, y_init, x_final, y_final, width):
    A = np.identity(3)
    drift = float(y_final - y_init)
    # We implicitly multiply by -1 if the order of the images is swapped...
    length = float(x_final - x_init)
    A[0, 2] = -0.5 * width
    # Negative because positive y points downwards
    A[1, 0] = -drift / length

    return A


def blendImages(ipv, blendWidth, is360=False, A_out=None):
    """
       INPUT:
         ipv: list of input images and their relative positions in the mosaic
         blendWidth: width of the blending function
       OUTPUT:
         croppedImage: final mosaic created by blending all images and
         correcting for any vertical drift
    """
    accWidth, accHeight, channels, width, translation = getAccSize(ipv)
    acc = pasteImages(
        ipv, translation, blendWidth, accWidth, accHeight, channels
    )
    compImage = normalizeBlend(acc)

    # Determine the final image width
    outputWidth = (accWidth - width) if is360 else accWidth
    # print("(accWidth - width)", (accWidth - width))
    # print("accWidth", accWidth)
    x_init, y_init, x_final, y_final = getDriftParams(ipv, translation, width)
    print(x_init, y_init, x_final, y_final)
    # Compute the affine transform
    A = np.identity(3)
    # BEGIN TODO 12
    # fill in appropriate entries in A to trim the left edge and
    # to take out the vertical drift if this is a 360 panorama
    # (i.e. is360 is true)
    # Shift it left by the correct amount
    # Then handle the vertical drift
    # Note: warpPerspective does forward mapping which means A is an affine
    # transform that maps accumulator coordinates to final panorama coordinates
    # TODO-BLOCK-BEGIN
    # print("start TODO 12  blendImages")

    if is360:
        A = computeDrift(x_init, y_init, x_final, y_final, width)

    # TODO-BLOCK-END
    # END TODO

    if A_out is not None:
        A_out[:] = A

    # Warp and crop the composite
    croppedImage = cv2.warpPerspective(
        compImage, A, (outputWidth, accHeight), flags=cv2.INTER_LINEAR
    )

    return croppedImage
