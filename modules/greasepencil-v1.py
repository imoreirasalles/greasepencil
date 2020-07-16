import os
import subprocess
import sys

# def install(package):
#    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# install("scikit-image")
import numpy as np
from skimage import exposure
from skimage import io
from skimage import img_as_ubyte
from skimage.filters import threshold_otsu, threshold_local
from skimage.measure import label, regionprops
from skimage.morphology import (
    closing,
    opening,
    square,
    remove_small_objects,
    erosion,
    dilation,
)
from skimage.segmentation import clear_border
from skimage.util import invert
from skimage.color import rgb2hsv, label2rgb
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

source = "/Users/martimpassos/dev/greasepencil/_input/"
destination = "/Users/martimpassos/dev/greasepencil/_output"


def make_image(data, outputname, size=(1, 1), dpi=80):
    fig = plt.figure()
    fig.set_size_inches(size)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(data, aspect="equal")
    plt.savefig(outputname, dpi=dpi)


def trim_black_border(img, tol=0):
    # img is 2D or 3D image data
    # tol  is tolerance
    mask = img > tol
    if img.ndim == 3:
        mask = mask.all(2)
    m, n = mask.shape
    mask0, mask1 = mask.any(0), mask.any(1)

    col_start = mask0.argmax()
    col_end = n - mask0[::-1].argmax()
    row_start = mask1.argmax()
    row_end = m - mask1[::-1].argmax()

    return img[row_start:row_end, col_start:col_end]


# define cropper function
def greasepencil(input_image):

    os.chdir(f"{source}")

    # import and preprocess source image
    original_image = img_as_ubyte(io.imread(input_image))
    # logarithmic_corrected = exposure.adjust_log(original_image, 2.1)
    image_bw = img_as_ubyte(io.imread(input_image, as_gray=True))
    # hsv_image = rgb2hsv(original_image)
    # image_bw = hsv_image[:,:,2]
    # image_bw = trim_black_border(image_bw, tol=50)
    # image_bw = invert(image_bw)

    # threshold and clean binary image
    thresh = threshold_otsu(image_bw)
    label_thresh = label(image_bw > thresh)
    mask = label_thresh == 1
    image_bw[mask] = 0
    # image_bw = erosion(image_bw)
    binary = image_bw > thresh / 3
    binary = erosion(binary)
    clean = remove_small_objects(binary, 2000)
    cleaner = closing(clean, square(99))
    # cleaner = remove_small_objects(clean, 15000)

    # label sub images
    label_image = label(cleaner)
    # mask = label_image == 1
    # label_image[mask] = 0
    fig, ax = plt.subplots()
    ax.imshow(label_image)
    ax.set_axis_off()
    region_list = []
    for region in regionprops(label_image):
        if 400000 > region.area >= 100000:
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle(
                (minc, minr),
                maxc - minc,
                maxr - minr,
                fill=False,
                edgecolor="red",
                linewidth=2,
            )
            ax.add_patch(rect)
            # region_list.append(coords)
    plt.tight_layout()
    # plt.savefig("/Users/martimpassos/Pictures/Artigo Greasepencil/thresh_opening73_inv.png",bbox_inches='tight')
    plt.show()
    print(f"A imagem {input_image} foi dividida em {len(region_list)} partes")

    """create output directory
    if not os.path.exists("../_output"):
        os.mkdir("../_output")
    os.chdir("..")"""

    # order sub images from top-left to bottom-right
    def origin_coord(item):
        return (item[0] * 4 + item[1]) / 5

    # get sub image coordinates
    region_list.sort(key=origin_coord)
    review_count = 0
    for region in region_list:
        counter = region_list.index(region) + 1
        height = region[2] - region[0]
        width = region[3] - region[1]
        ratio = width / height
        if ratio < 1:
            ratio = 1 / ratio

        # save sub image and name it accordingly
        if 1.4 < ratio < 1.61:
            io.imsave(
                f"{destination}/{input_image[0:-4]}-{counter:02}.jpg",
                original_image[region[0] : region[2], region[1] : region[3]],
                quality=100,
            )

        else:
            io.imsave(
                f"{destination}/{input_image[0:-4]}-{counter:02}-REVISAR.jpg",
                original_image[region[0] : region[2], region[1] : region[3]],
                quality=100,
            )
            review_count += 1

    print(f"Encontrei {review_count} problemas na imagem {input_image}")


# list files to be processed
extensions = (".jpg", ".jpeg")
image_files = [
    image_file
    for image_file in os.listdir(f"{source}")
    if str(image_file).endswith(tuple(extensions))
]

# call function
# for i in image_files:
greasepencil("/Users/martimpassos/dev/greasepencil/_input/01.jpg")
# print(f"Você forneceu {len(image_files)} imagens e gerou {len(os.listdir(destination))} arquivos!")
