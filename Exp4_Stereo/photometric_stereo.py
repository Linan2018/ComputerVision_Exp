import numpy as np
import math
import sys

#from scipy.misc import imsave
from imageio import imsave

from dataset import load_dataset

from util import compute_photometric_stereo, rerendering_error

assert len(sys.argv) > 1
data = load_dataset(sys.argv[1])
# print(type(data.lights), type(data.right), isinstance(data.right, list))

assert isinstance(data.right, list)
# 有些数据集data.right是一个python3的迭代对象，需要保证他为一个list
# 最简单但是并不保险的方法，是进行以下修改： data.right = list(data.right)

albedo, normals = compute_photometric_stereo(data.lights, data.right)

avg_rmse = rerendering_error(data.lights, data.right, albedo, normals)

print('Average RMSE of rerendered image is {0}'.format(avg_rmse))

print('Saving albedo to {0}'.format(data.albedo_png))
imsave(data.albedo_png, np.uint8(np.clip(albedo, 0, 255)))
print('Saving normals visualization to {0}'.format(data.normals_png))
imsave(data.normals_png, normals)
print('Saving normals to {0}'.format(data.normals_npy))
np.save(data.normals_npy, normals)
