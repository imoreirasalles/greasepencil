import os
import glob
from skimage import io
from skimage import img_as_ubyte
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square

# collection = img_as_ubyte(io.ImageCollection("*.jpg"))


def cropper(input_image):
    original_img = img_as_ubyte(io.imread(input_image))
    image_bw = original_img[:, :, 1]
    # apply threshold
    thresh = threshold_otsu(image_bw)
    # thresh = threshold_multiotsu(image_bw)
    bw = closing(image_bw > thresh / 5, square(13))

    # remove artifacts connected to image border
    cleared = clear_border(bw)

    # label image regions
    label_image = label(cleared)

    if not os.path.exists("./_output"):
        os.mkdir("_output")

    count = 1
    rename_map = {1: 4, 2: 1, 3: 3, 4: 2}
    # rename_map = {k + 4: v + 4 for k, v in rename_map.items()}
    # print(rename_map)
    # if (count - 1) %4 == 0

    for region in regionprops(label_image):

        # take regions with large enough areas
        if region.area >= 20000:
            minr, minc, maxr, maxc = region.bbox
            ratio = (maxc - minc) / (maxr - minr)
            if ratio < 1:
                ratio = 1 / ratio
            # save large enough regions to files
            if 1.4 <= ratio <= 1.61:
                image_name = input_image[0:-4] + "_" + str(rename_map[count])
            else:
                image_name = input_image[0:-4] + "_" + str(rename_map[count]) + "_ERRO"
            io.imsave(f"./_output/{image_name}.jpg", original_img[minr:maxr, minc:maxc])
            count += 1


# for i in collection:
for filepath in glob.iglob("./_input/*jpg", recursive=True):
    cropper(filepath)

