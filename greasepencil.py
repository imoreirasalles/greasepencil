import os
from skimage import exposure
from skimage import io
from skimage import img_as_ubyte
from skimage.filters import threshold_otsu, threshold_local
from skimage.measure import label, regionprops
from skimage.morphology import closing, square, remove_small_objects
from skimage.segmentation import clear_border
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# define cropper function
def greasepencil(input_image):

    os.chdir("./_input")

    # import and preprocess source image
    original_image = img_as_ubyte(io.imread(input_image))
    logarithmic_corrected = exposure.adjust_log(original_image, 2.1)
    image_bw = logarithmic_corrected[:, :, 2]

    # threshold and clean binary image
    thresh = threshold_otsu(image_bw)
    clean = closing(image_bw > thresh, square(75))
    cleaner = remove_small_objects(clean, 10000)

    # label sub images
    label_image = label(cleaner)
    mask = label_image == 1
    label_image[mask] = 0
    region_list = []
    for region in regionprops(label_image):
        if 400000 > region.area >= 100000:
            coords = region.bbox
            region_list.append(coords)


    # create output directory
    if not os.path.exists("../_output"):
        os.mkdir("../_output")
    os.chdir("..")

    # order sub image from top-left to bottom-right
    def origin_coord(item):
        return (item[0] * 4 + item[1]) / 5

    # get sub image coordinates
    region_list.sort(key=origin_coord)
    for region in region_list:
        counter = region_list.index(region) + 1
        height = region[2] - region[0]
        width = region[3] - region [1]
        ratio = width/height
        if ratio < 1:
            ratio = 1/ratio

    # save sub image and name it accordingly       
        if 1.4 < ratio < 1.61:
            io.imsave(
                f"./_output/{input_image[0:-4]}-{counter:02}.jpg",
                original_image[region[0] : region[2], region[1] : region[3]], quality=100
            )
        else:
            io.imsave(
                f"./_output/{input_image[0:-4]}-{counter:02}-REVISAR.jpg",
                original_image[region[0] : region[2], region[1] : region[3]], quality=100
            )

# list files to be processed
image_files = [
    image_file
    for image_file in os.listdir("./_input")
    if str(image_file).endswith(".jpg")
]

# call function
for i in image_files:
    greasepencil(i)
