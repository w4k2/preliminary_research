import numpy as np
from scipy import signal
import png
import time
from matplotlib.pyplot import imread
from scipy.ndimage import gaussian_filter
from scipy.ndimage import grey_erosion, grey_dilation
import cv2

# Definitions
D = 2                   # Universe dimensions [Please stay in 2D]
N = 1080                  # Quant per dimension
void = 1            # Void strength
fps = 25               # Frames per second
eternity = fps * 15     # Iteration limit
rs = 1410                 # World number
gaia = False            # Gaia or Medeja hypothesis
comsize = 3             # Center of mind size
godfinger = (1080//2,1080//2)     # Universe startingpoint

# Carefully declare frequency of mind
#  9 — MESSAGE FROM SPACE
#  5 — EPILEPSY
# .9 — DEEP SLEEP
# .5 — REM
# .2 — DESERT WANDERER
#  0 — MINDLESS
fom = 1
fmind = np.random.normal(size=[2*comsize for _ in range(D)])*fom

# Get filename
filename = "videos/%s-D%i-N%i-V010%i-C%i-v%05i-FOM%3.3f.mp4" % (
    "GAIA" if gaia else "MEDEJA",
    D, N, void*1000000, comsize,rs,
    fom
)

# Helpers
out = cv2.VideoWriter(filename,
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      fps, (N, N), False)

def whereismy(mind):
    img = np.copy(mind)
    img -= np.min(img)
    img /= np.max(img)
    img = (img*255).astype(np.uint8)
    png.from_array(img, 'L').save("foo.png")
    #time.sleep(1/fps)
    out.write(img)


# Create D-dimensional mind of N quants
mind = np.zeros([N for _ in range(D)])

# Touch of creation
if rs is not None:
    np.random.seed(rs)
else:
    rs = -1
mind[godfinger] = 1
energy = 200

for i in range(fps//2):
    whereismy(mind)

# Iterate the world's mind
for i in range(eternity):
    # Establish normalized center of mind
    com = mind[godfinger[0]-comsize:godfinger[0]+comsize,
               godfinger[1]-comsize:godfinger[1]+comsize]
    com -= np.min(com)
    com /= np.max(com)

    # Change mind by closed correlation
    mind = signal.correlate2d(mind, com+fmind, mode='same', boundary='wrap')
    mind = grey_dilation(mind,footprint=com, mode='wrap')
    mind = grey_erosion(mind,footprint=com, mode='wrap')

    # Measure energy
    _energy = np.sum(mind)

    # Gaia-Medeja dychotomy
    if gaia:  # Gaia
        goddess = 0
        r = 1
        medeja_strength = 0
    else:   # Medeja
        # Prepare mindless destruction
        goddess = np.random.normal(size=mind.shape)
        medeja_strength = np.sum(goddess)

        # Calculate anger ratio r
        r = (energy-_energy)/medeja_strength


    # Destruct and normalize
    mind += goddess / r
    mind /= np.max(np.abs(mind))

    #if i > 1:
    # Observe state
    whereismy(mind)
    if gaia:
        print("GAIA:%i-%i-%i | Y %010i | Energy: %10.3f" % (
            rs, D, N,
            i,_energy))
    else:
        print("MEDEJA:%i-%i-%i | Y %010i | Energy: %10.3f | Medeja: %05i %%" % (
            rs, D, N,
            i,_energy,_energy/np.abs(medeja_strength)))

out.release()
