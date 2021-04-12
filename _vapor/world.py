import numpy as np
from scipy import signal
import png
import time
from matplotlib.pyplot import imread
from scipy.ndimage import gaussian_filter
from scipy.ndimage import grey_erosion, grey_dilation
import cv2

# Definitions
D = 2                       # Universe dimensions [Please stay in 2D]
N = 256                     # Quant per dimension
void = 0.0000000001#.1      # Void strength
fps = 25                    # Frames per second
eternity = fps * 10         # Iteration limit
rs = np.random.randint(1410)                   # World number
gaia = False                 # Gaia or Medeja hypothesis
comsize = 1                 # Center of mind size
godfinger = (N//2,N//2)     # Universe startingpoint
gaia_counter = 1 * 1
v = 0.00001
intro = 0
energy = .0

# Carefully declare frequency of mind
#  9 — MESSAGE FROM SPACE
#  5 — EPILEPSY
# .9 — DEEP SLEEP
# .5 — REM
# .2 — DESERT WANDERER
#  0 — MINDLESS
fom = 1    # Spoistość może się zerwać przy v
#fmind = np.random.normal(size=[2*comsize for _ in range(D)])*fom

# Get filename
filename = "videos/%s-D%i-N%i-V010%i-C%i-v%05i-FOM%3.3f.mp4" % (
    "GAIA" if gaia else "MEDEJA",
    D, N, void*1000000, comsize,rs,
    fom
)

# Helpers
out = cv2.VideoWriter(filename,
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      fps, (N, N), True)

def wherearemy(minds):
    imgs = np.copy(minds)
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
minds = np.zeros((*[N for _ in range(D)], 3))

# Touch of creation
if rs is not None:
    np.random.seed(rs)
else:
    rs = -1

minds[godfinger] = 1


for i in range(intro):
    whereismy(mind, mind2, mind3)

# Iterate the world's mind
for i in range(eternity):
    # Changing fmind
    fmind = np.random.normal(size=(*[2*comsize+1
                                    for _ in range(D)], 3)) * fom

    print("FS", fmind.shape)


    # Establish normalized center of mind
    com = minds[godfinger[0]-comsize-1:godfinger[0]+comsize,
                godfinger[1]-comsize-1:godfinger[1]+comsize]
    com -= np.min(com)
    com /= np.max(com)

    # Change mind by closed correlation
    for channel in range(3):
        minds[:,:,channel] = signal.correlate2d(minds[:,:,channel],
                                                com[:,:,channel] + fmind[:,:,channel],
                                                mode='same',
                                                boundary='wrap')
        #if np.random.randint(skip)!=0:
        minds[:,:,channel] = grey_dilation(minds[:,:,channel],
                                           footprint=com[:,:,channel],
                                           mode='wrap')
        minds[:,:,channel] = grey_erosion(minds[:,:,channel],
                                           footprint=com[:,:,channel],
                                           mode='wrap')
        #if np.random.randint(skip)!=0:
        #    minds = grey_erosion(minds,footprint=com, mode='wrap')

    # Measure energy
    _energy = np.sum(minds)

    # Prepare mindless destruction
    goddess = np.random.normal(size=minds.shape) * energy

    # Destruct and normalize
    minds += goddess
    minds /= np.max(np.abs(minds))
    #minds[godfinger] = np.random.normal(size=3)

    # Observe state
    wherearemy(minds)

out.release()
