import numpy as np
from scipy import signal
import png
import time
from matplotlib.pyplot import imread
from scipy.ndimage import gaussian_filter
from scipy.ndimage import grey_erosion, grey_dilation
import cv2

# Definitions
N = 128                     # Quant per dimension
fps = 25                    # Frames per second
r = 2
eternity = fps * 10         # Iteration limit
rs = np.random.randint(1410)                   # World number
zoom = 1
# Get filename
filename = "video.mp4"

# Helpers
out = cv2.VideoWriter(filename,
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      fps, (N, N), True)

def whereismy(mind):
    imgs = np.copy(mind)
    imgs -= np.min(imgs)
    imgs /= np.max(imgs)
    imgs = (imgs*255).astype(np.uint8)

    # Prepare for Video
    cimg = np.copy(imgs)
    out.write(np.flip(cimg, 2))

    # Prepare for PNG
    ccimg = np.copy(cimg)
    ccimg = ccimg.reshape(ccimg.shape[0], -1).astype(np.uint8).copy()
    png.from_array(ccimg, 'RGB').save("foo.png")
    time.sleep(1/fps)

# Create D-dimensional mind of N quants
mind = np.random.normal(size=(N, N, 3))
mind = np.zeros((N, N, 3))
mind[N//2, N//2] = 1

# Iterate the world's mind
for i in range(eternity):
    # Changing mind
    com = mind[N // 2 - r:N // 2 + r,
               N // 2 - r:N // 2 + r] + 1
    fcom = np.random.normal(size=com.shape) * zoom
    com = (fcom + com) / 2
    com[1,1] = np.random.normal()

    for ch in range(3):
        a = ch
        b = ch + 1
        if b == 3:
            b = 0
        mind[:,:,ch] = signal.correlate2d(mind[:,:,a],
                                          com[:,:,b],
                                          mode='same',
                                          boundary='wrap')
        mind[:,:,ch] = grey_dilation(mind[:,:,a],
                                     footprint=com[:,:,b],
                                     mode='wrap')
        mind[:,:,ch] = grey_erosion(mind[:,:,a],
                                    footprint=com[:,:,a],
                                    mode='wrap')
        mind[:,:,ch] = signal.correlate2d(mind[:,:,a],
                                          com[:,:,b],
                                          mode='same',
                                          boundary='wrap')

    mind -= np.min(mind)
    mind /= np.max(mind)
    # Observe state
    whereismy(mind)

out.release()
