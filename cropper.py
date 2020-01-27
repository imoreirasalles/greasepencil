import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

from skimage import io
from skimage import img_as_ubyte
from skimage.filters import (
    threshold_otsu,
    rank,
    sobel,
    threshold_niblack,
    threshold_sauvola,
    threshold_li,
    threshold_local,
    threshold_minimum,
    threshold_multiotsu,
)
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square, rectangle, convex_hull_image
from skimage.color import label2rgb
from skimage.morphology import disk
from skimage.transform import rescale
from skimage.util import invert


imagem = img_as_ubyte(io.imread("01.jpg",))
# collection = img_as_ubyte(io.ImageCollection("*.jpg"))


def cropper(source):
    image_bw = img_as_ubyte(source[:, :, 1])
    # apply threshold
    thresh = threshold_otsu(image_bw)
    # thresh = threshold_multiotsu(image_bw)
    bw = closing(image_bw > thresh / 3.5, square(15))

    # remove artifacts connected to image border
    cleared = clear_border(bw)

    # label image regions
    label_image = label(cleared)

    if not os.path.exists("./OUTPUT"):
        os.mkdir("OUTPUT")
    count = 21

    for region in regionprops(label_image):
        # take regions with large enough areas
        if region.area >= 20000:
            minr, minc, maxr, maxc = region.bbox
            ratio = (maxc - minc) / (maxr - minr)e
            # save large enough regions to files
            if 1.45 <= ratio <= 1.6:
                image_name = "image" + str(count)
            else:
                image_name = "image" + str(count) + "_ERRO"
            io.imsave(
                f"./OUTPUT/{image_name}.jpg", source[minr:maxr, minc:maxc],
            )
            count += 1


# for i in collection:
cropper(imagem)
