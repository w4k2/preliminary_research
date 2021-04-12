import numpy as np
from scipy import signal
import png
import time
from matplotlib.pyplot import imread
from scipy.ndimage import gaussian_filter
from scipy.ndimage import grey_erosion, grey_dilation
from skimage.transform import rescale, resize
import cv2
from PIL import Image, ImageFont, ImageDraw
import datetime



rs = int(datetime.datetime.now().timestamp())
np.random.seed(rs)
fpss = [12, 24, 25, 50, 60]
fpss = [12]
Ns = [16,32,64,128,256, 512, 1024, 2048]
Ns = [8]
# Definitions
D = 2                   # Universe dimensions [Please stay in 2D]
fps = np.random.choice(fpss)               # Frames per second
gaia = True            # Gaia or Medeja hypothesis

N = np.random.choice(Ns)                  # Quant per dimension
godfinger = (N//2,N//2)

dyno = np.random.randint(0,2) == 0
additive = np.random.randint(0,2) == 0
# TOSS
void = np.abs(np.random.normal(0, .5))#.1            # Void strength
eternity = fps * 2 # np.random.randint(5,10)     # Iteration limit
comsize = np.random.randint(1,3)             # Center of mind size
gaia_counter = fps * np.random.randint(0,30)
mono = np.random.randint(0, 10)
v = np.abs(np.random.normal(0, .5))
fom = np.abs(np.random.normal(0, .5))    # Spoistość może się zerwać przy v


# Carefully declare frequency of mind
#  9 — MESSAGE FROM SPACE
#  5 — EPILEPSY
# .9 — DEEP SLEEP
# .5 — REM
# .2 — DESERT WANDERER
#  0 — MINDLESS
fmind = np.random.normal(size=[2*comsize for _ in range(D)])*fom

def text_phantom(text, size, c=(0,0,0)):
    # Availability is platform dependent

    # Create font
    pil_font = ImageFont.truetype("C64.ttf", size=16,
                                  encoding="unic")
    text_width, text_height = pil_font.getsize(text)

    # create a blank canvas with extra space between lines
    canvas = Image.new('RGB', [size, size], (255, 255, 255))

    # draw the text onto the canvas
    draw = ImageDraw.Draw(canvas)
    offset = ((size - text_width) // 2,
              (size - text_height) // 2)
    draw.text((1024//16+4,1024//16+8), text, font=pil_font, fill=tuple(c))

    # Convert the canvas into an array with values in [0, 1]
    return (255 - np.asarray(canvas)) / 255.0

print("FOM%.3f" % fom, "T", eternity//fps, "GC", gaia_counter//fps, "N", N, "RS", rs, "CS", comsize)


# Get filename
filename = "videos/%i.mp4" % (rs)
pngfilename = "thumbnails/%i.png" % (rs)
colfilename = "szymel/_simulations/%i.md" % (rs)

# Save to collection
file1 = open(colfilename,"w")
L = ["---\n","rs: %i\n" % rs,"---\n"]
file1.writelines(L)
file1.close() #to change file access modes

# Helpers
out = cv2.VideoWriter(filename,
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      fps, (1024, 1024), True)

def whereismy(mind, mind2, mind3, ii, ee):
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

    #print(img.shape)

    # Prepare for Video
    if mono == 1:
        mimg = img//3 + img2//3 + img3//3
        cimg = np.concatenate(
            (mimg[:,:,np.newaxis],
             mimg[:,:,np.newaxis],
             mimg[:,:,np.newaxis]),
            axis=2
        ).astype(np.uint8)
    elif mono == 0:
        mimg = img + img2 + img3
        mimg = mimg // 3
        cimg = np.concatenate(
            (mimg[:,:,np.newaxis],
             mimg[:,:,np.newaxis],
             mimg[:,:,np.newaxis]),
            axis=2
        ).astype(np.uint8)
    elif mono == 2:
        mimg = img//3 + img2//3 + img3//3
        cimg = np.concatenate(
            (img[:,:,np.newaxis],
             mimg[:,:,np.newaxis],
             mimg[:,:,np.newaxis]),
            axis=2
        ).astype(np.uint8)

    else:
        cimg = np.concatenate(
            (img[:,:,np.newaxis],
             img2[:,:,np.newaxis],
             img3[:,:,np.newaxis]),
            axis=2
        ).astype(np.uint8)
    #print(cimg)
    #print("CS",cimg.shape)

    #print(ttp, ttp.shape)

    meanc = (np.mean(np.mean(cimg, axis=0), axis=0)).astype(int)

    print(meanc)
    #meanc = (255,255,0)

    ttp = text_phantom(text_ttp, 1024, meanc)

    aaa = resize(cimg, (1024, 1024), anti_aliasing=False, order=0)


    cimg = (255*np.clip((aaa), 0, 1)).astype(np.uint8)
    #print(cimg)

    cimg[ttp!=0] = 255


    print(ii, ee, ii/ee, 1024//N)

    cimg[-8:,:int((ii+1)*1024/ee)] = 255#-meanc

    #exit()

    #print("CS",cimg.shape)
    out.write(np.flip(cimg, 2))

    # Prepare for PNG
    ccimg = np.copy(cimg)
    ccimg = ccimg.reshape(ccimg.shape[0], -1).astype(np.uint8).copy()
    png.from_array(ccimg, 'RGB').save("foo.png")
    png.from_array(ccimg, 'RGB').save(pngfilename)
    time.sleep(1/fps)


# Create D-dimensional mind of N quants
mind = np.zeros([N for _ in range(D)])
mind2 = np.zeros([N for _ in range(D)])
mind3 = np.zeros([N for _ in range(D)])

# Touch of creation

"""
mind[godfinger[0]-comsize:godfinger[0]+comsize,
     godfinger[1]-comsize:godfinger[1]+comsize] = 1
mind2[godfinger[0]-comsize:godfinger[0]+comsize,
      godfinger[1]-comsize:godfinger[1]+comsize] = 1
mind3[godfinger[0]-comsize:godfinger[0]+comsize,
      godfinger[1]-comsize:godfinger[1]+comsize] = 1
"""

mind[godfinger] = np.random.randint(255)
mind2[godfinger] = np.random.randint(255)
mind3[godfinger] = np.random.randint(255)

mind[godfinger] = 1
mind2[godfinger] = 1
mind3[godfinger] = 1
mind[godfinger[0]+1,godfinger[1]+1] = 1
mind2[godfinger[0]-1,godfinger[1]-1] = 1
mind3[godfinger[0]-1,godfinger[1]+1] = 1

energy = 2

# Iterate the world's mind
for i in range(eternity):
    if dyno:
        if i%fps == 0:
            #eternity = fps * np.random.randint(5,45)     # Iteration limit
            comsize = np.random.randint(1,3)             # Center of mind size
            #gaia_counter = fps * np.random.randint(0,30)
            mono = np.random.randint(0, 5)
            if additive:
                void += np.abs(np.random.normal(0, .5))#.1            # Void strength
                fom += np.abs(np.random.normal(0, .5))    # Spoistość może się zerwać przy v

            else:
                void = np.abs(np.random.normal(0, .5))#.1            # Void strength
                fom = np.abs(np.random.normal(0, .5))    # Spoistość może się zerwać przy v


            v = np.abs(np.random.normal(0, .5))


    text_ttp = "♥ %.5f-%s-%s%s\n♣ %i\n\nF%.3f/V%.3f\nT%i/GC%i\nN%i/CS%i/M%i\n%iFPS" % (
        (i+1)/eternity, 'Gaia' if gaia else 'Medea', 'D' if dyno else '-', 'A' if additive else '-', rs,
        fom, void, eternity//fps, gaia_counter//fps,N,comsize,mono,
        fps
    )

    # Changing fmind
    fmind = np.random.normal(size=[2*comsize+1 for _ in range(D)]) * fom


    # Establish normalized center of mind
    com = mind[godfinger[0]-comsize-1:godfinger[0]+comsize,
               godfinger[1]-comsize-1:godfinger[1]+comsize]
    com -= np.min(com)
    com /= np.max(com)

    com2 = mind2[godfinger[0]-comsize-1:godfinger[0]+comsize,
               godfinger[1]-comsize-1:godfinger[1]+comsize]
    com2 -= np.min(com2)
    com2 /= np.max(com2)

    com3 = mind3[godfinger[0]-comsize-1:godfinger[0]+comsize,
               godfinger[1]-comsize-1:godfinger[1]+comsize]
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
    if not gaia:
        #print(com.shape, fmind.shape)
        rr = comsize*2+1
        a = np.random.randint(0, N-rr, size=2)


        mind += signal.correlate2d(mind, mind[a[0]:a[0]+rr,
                                             a[1]:a[1]+rr]+fmind, mode='same', boundary='wrap')*void
        mind2 += signal.correlate2d(mind2, mind2[a[0]:a[0]+rr,
                                               a[1]:a[1]+rr]+fmind, mode='same', boundary='wrap')*void
        mind3 += signal.correlate2d(mind3, mind3[a[0]:a[0]+rr,
                                               a[1]:a[1]+rr]+fmind, mode='same', boundary='wrap')*void
        fp = mind[a[0]:a[0]+rr,a[1]:a[1]+rr]

        mind[a[0]+1,a[1]+1] = np.random.normal(np.mean(mind), np.std(mind)*v+void)
        mind2[a[0]+1,a[1]+1] = np.random.normal(np.mean(mind2), np.std(mind2)*v+void)
        mind3[a[0]+1,a[1]+1] = np.random.normal(np.mean(mind3), np.std(mind3)*v+void)

        if np.sum(fp) > 1:
            mind +=grey_dilation(mind,footprint=fp, mode='wrap')*void
            mind += grey_erosion(mind,footprint=fp, mode='wrap')*void
            mind2 += grey_dilation(mind2,footprint=fp, mode='wrap')*void
            mind2 += grey_erosion(mind2,footprint=fp, mode='wrap')*void
            mind3 += grey_dilation(mind3,footprint=fp, mode='wrap')*void
            mind3 += grey_erosion(mind3,footprint=fp, mode='wrap')*void


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
    whereismy(mind, mind2, mind3, i, eternity)

out.release()
