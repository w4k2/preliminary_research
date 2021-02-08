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
N = 1080//8                  # Quant per dimension
#N = 512                  # Quant per dimension
void = .1            # Void strength
fps = 25               # Frames per second
eternity = fps * 500     # Iteration limit
rs = 1410                 # World number
gaia = True            # Gaia or Medeja hypothesis
comsize = 2             # Center of mind size
godfinger = (1080//16,1080//16)     # Universe startingpoint
#godfinger = (255,255)
gaia_counter = fps * 1

# Input
#input = imread('lena.png')
# print(input.shape)


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
                      fps, (N, N), True)

def whereismy(mind, mind2, mind3):
    img = np.copy(mind)
    img -= np.min(img)
    img /= np.max(img)
    img = (img*255).astype(np.uint8)

    img2 = np.copy(mind2)
    img2 -= np.min(img2)
    img2 /= np.max(img2)
    img2 = (img2*255).astype(np.uint8)


    img3 = np.copy(mind3)
    img3 -= np.min(img3)
    img3 /= np.max(img3)
    img3 = (img3*255).astype(np.uint8)

    print(img.shape)

    # Prepare for Video
    cimg = np.concatenate(
        (img[:,:,np.newaxis],
         img2[:,:,np.newaxis],
         img3[:,:,np.newaxis]),
        axis=2
    ).astype(np.uint8)
    out.write(np.flip(cimg, 2))

    # Prepare for PNG
    ccimg = np.copy(cimg)
    ccimg = ccimg.reshape(ccimg.shape[0], -1).astype(np.uint8).copy()
    png.from_array(ccimg, 'RGB').save("foo.png")
    time.sleep(1/fps)


# Create D-dimensional mind of N quants
mind = np.zeros([N for _ in range(D)])
mind2 = np.zeros([N for _ in range(D)])
mind3 = np.zeros([N for _ in range(D)])

# Touch of creation
if rs is not None:
    np.random.seed(rs)
else:
    rs = -1

mind[godfinger[0]-comsize:godfinger[0]+comsize,
     godfinger[1]-comsize:godfinger[1]+comsize] = 1
mind2[godfinger[0]-comsize:godfinger[0]+comsize,
      godfinger[1]-comsize:godfinger[1]+comsize] = 1
mind3[godfinger[0]-comsize:godfinger[0]+comsize,
      godfinger[1]-comsize:godfinger[1]+comsize] = 1

mind[godfinger] = 0
mind2[godfinger] = 0
mind3[godfinger] = 0

energy = 200

for i in range(fps//2):
    whereismy(mind, mind2, mind3)

# Iterate the world's mind
for i in range(eternity):
    # Establish normalized center of mind
    com = mind[godfinger[0]-comsize:godfinger[0]+comsize,
               godfinger[1]-comsize:godfinger[1]+comsize]
    com -= np.min(com)
    com /= np.max(com)

    com2 = mind2[godfinger[0]-comsize:godfinger[0]+comsize,
               godfinger[1]-comsize:godfinger[1]+comsize]
    com2 -= np.min(com2)
    com2 /= np.max(com2)

    com3 = mind3[godfinger[0]-comsize:godfinger[0]+comsize,
               godfinger[1]-comsize:godfinger[1]+comsize]
    com3 -= np.min(com3)
    com3 /= np.max(com3)

    #com = np.mean([com, com2, com3], axis=0)


    # Change mind by closed correlation
    mind = signal.correlate2d(mind, com+fmind, mode='same', boundary='wrap')
    mind = grey_dilation(mind,footprint=com, mode='wrap')
    mind = grey_erosion(mind,footprint=com, mode='wrap')

    mind2 = signal.correlate2d(mind2, com2+fmind, mode='same', boundary='wrap')
    mind2 = grey_dilation(mind2,footprint=com2, mode='wrap')
    mind2 = grey_erosion(mind2,footprint=com2, mode='wrap')

    mind3 = signal.correlate2d(mind3, com3+fmind, mode='same', boundary='wrap')
    mind3 = grey_dilation(mind3,footprint=com3, mode='wrap')
    mind3 = grey_erosion(mind3,footprint=com3, mode='wrap')

    # Extramass
    """
    print(com.shape, fmind.shape)
    rr = 4
    a = np.random.randint(0, N-rr, size=2)

    print(a)


    mind = signal.correlate2d(mind, mind[a[0]:a[0]+rr,
                                         a[1]:a[1]+rr]+fmind, mode='same', boundary='wrap')
    mind2 = signal.correlate2d(mind2, mind[a[0]:a[0]+rr,
                                           a[1]:a[1]+rr]+fmind, mode='same', boundary='wrap')
    mind3 = signal.correlate2d(mind3, mind[a[0]:a[0]+rr,
                                           a[1]:a[1]+rr]+fmind, mode='same', boundary='wrap')
    fp = mind[a[0]:a[0]+rr,a[1]:a[1]+rr]
    """


    """
    if np.sum(fp) > 0:
        mind = grey_dilation(mind,footprint=fp, mode='wrap')
        mind = grey_erosion(mind,footprint=fp, mode='wrap')
        mind2 = grey_dilation(mind2,footprint=fp, mode='wrap')
        mind2 = grey_erosion(mind2,footprint=fp, mode='wrap')
        mind3 = grey_dilation(mind3,footprint=fp, mode='wrap')
        mind3 = grey_erosion(mind3,footprint=fp, mode='wrap')
    """

    #mind = grey_dilation(mind,footprint=com, mode='wrap')
    #mind = grey_erosion(mind,footprint=com, mode='wrap')


    # Measure energy
    _energy = np.sum(mind)
    _energy2 = np.sum(mind2)
    _energy3 = np.sum(mind3)

    if i > gaia_counter:
        gaia = False

    #gaia = i%2

    # Gaia-Medeja dychotomy
    if gaia:  # Gaia
        goddess = 0
        goddess2 = 0
        goddess3 = 0
        r = 1
        r2 = 1
        r3 = 1
        medeja_strength = 0
    else:   # Medeja
        # Prepare mindless destruction
        goddess = np.random.normal(size=mind.shape)
        medeja_strength = np.sum(goddess)

        goddess2 = np.random.normal(size=mind2.shape)
        medeja_strength2 = np.sum(goddess2)

        goddess3 = np.random.normal(size=mind3.shape)
        medeja_strength3 = np.sum(goddess3)

        # Calculate anger ratio r
        r = (energy-_energy)/medeja_strength
        r2 = (energy-_energy2)/medeja_strength2
        r3 = (energy-_energy3)/medeja_strength3


    # Add input image
    #mind += input[:,:,0]*.1
    #mind2 += input[:,:,1]*.1
    #mind3 += input[:,:,2]*.1

    # Destruct and normalize
    mind += goddess / r
    mind /= np.max(np.abs(mind))

    mind2 += goddess2 / r2
    mind2 /= np.max(np.abs(mind2))

    mind3 += goddess3 / r3
    mind3 /= np.max(np.abs(mind3))

    # Observe state
    whereismy(mind, mind2, mind3)

out.release()
