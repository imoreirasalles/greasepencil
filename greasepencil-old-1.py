import os
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install("scikit-image")

from skimage import exposure
from skimage import io
from skimage import img_as_ubyte
from skimage.filters import threshold_otsu, threshold_local
from skimage.measure import label, regionprops
from skimage.morphology import closing, square, remove_small_objects
from skimage.segmentation import clear_border
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

source = input("Em qual pasta estão as imagens matriz?")
destination = input("Em qual pasta deseja salvar as imagens individualizadas?")

# define cropper function
def greasepencil(input_image):

    os.chdir(f"{source}")

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
    print(f"A imagem {input_image} foi dividida em {len(region_list)} partes")


    '''create output directory
    if not os.path.exists("../_output"):
        os.mkdir("../_output")
    os.chdir("..")'''

    # order sub images from top-left to bottom-right
    def origin_coord(item):
        return (item[0] * 4 + item[1]) / 5

    # get sub image coordinates
    region_list.sort(key=origin_coord)
    review_count = 0
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
                f"{destination}/{input_image[0:-4]}-{counter:02}.jpg",
                original_image[region[0] : region[2], region[1] : region[3]], quality=100
            )
            
        else:
            io.imsave(
                f"{destination}/{input_image[0:-4]}-{counter:02}-REVISAR.jpg",
                original_image[region[0] : region[2], region[1] : region[3]], quality=100
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
for i in image_files:
    greasepencil(i)
print(f"Você forneceu {len(image_files)} imagens e gerou {len(os.listdir(destination))} arquivos!")
