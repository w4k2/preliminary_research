import numpy as np
from scipy import signal
import png
import time
from matplotlib.pyplot import imread
from scipy.ndimage import gaussian_filter
from scipy.ndimage import grey_erosion, grey_dilation

# Definitions
D = 2                   # Universe dimensions [Please stay in 2D]
N = 64                  # Quant per dimension
void = .0001            # Void strength
eternity = 100000000    # Iteration limit
fps = 25                # Frames per second
rs = None               # World number
gaia = False            # Gaia or Medeja hypothesis
comsize = 2             # Center of mind size
godfinger = (31,31)     # Universe startingpoint

# Carefully declare frequency of mind
#  9 — MESSAGE FROM SPACE
#  5 — EPILEPSY
# .9 — DEEP SLEEP
# .5 — REM
# .2 — DESERT WANDERER
#  0 — MINDLESS
fom = .5
fmind = np.random.normal(size=[2*comsize for _ in range(D)])*fom

# Helpers
def whereismy(mind):
    img = np.copy(mind)
    img -= np.min(img)
    img /= np.max(img)
    img = (img*255).astype(np.uint8)
    png.from_array(img, 'L').save("foo.png")
    time.sleep(1/fps)

# Create D-dimensional mind of N quants
mind = np.zeros([N for _ in range(D)])

# Touch of creation
if rs is not None:
    np.random.seed(rs)
else:
    rs = -1
mind[godfinger] = 1
energy = 200

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
