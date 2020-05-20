import subprocess
import os
import platform

#from scipy.misc import imsave
from imageio import imsave


class GifWriter(object):

    def __init__(self, temp_format, dest_gif):
        try:
            sysstr = platform.system()
            if sysstr =="Windows":
                # Windows上可能需要加上 `shell=True` 才能运行。有些时候又需要设置为`False`才能运行，取决于你的系统版本。40行左右同理。
                # Linux天下第一！别问为什么。
                subprocess.check_call(
                    ['magick'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            else:
                subprocess.check_call(
                    ['convert'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except OSError:
            raise Exception('imagemagick is required for gif support')
        except subprocess.CalledProcessError:
            pass

        self.temp_format = temp_format
        self.dest_gif = dest_gif
        self.temp_filenames = []
        self.closed = False

    def append(self, image):
        if self.closed:
            raise Exception('GifWriter is already closed')
        filename = self.temp_format % len(self.temp_filenames)
        self.temp_filenames.append(filename)
        imsave(filename, image)

    def close(self):
        sysstr = platform.system()
        if sysstr =="Windows":  # 此处同样进行了调整
            subprocess.check_call(['magick', '-delay', '2', '-loop', '0'] +
                              self.temp_filenames + [self.dest_gif], shell=True)
        else:
            subprocess.check_call(['convert', '-delay', '2', '-loop', '0'] +
                              self.temp_filenames + [self.dest_gif])
        for filename in self.temp_filenames:
            os.unlink(filename)
        self.closed = True
