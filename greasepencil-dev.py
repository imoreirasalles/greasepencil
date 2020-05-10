import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.feature import match_template, peak_local_max
from skimage.morphology import extrema
from skimage.measure import label
from skimage.segmentation import watershed
from skimage import exposure, transform, img_as_ubyte, io, exposure, color, feature

"""source = input("Em qual pasta estão as imagens matriz?")
destination = input("Em qual pasta deseja salvar as imagens individualizadas?")

class slide:
    def __init__(self, dimension):
        self.frame_bg = np.full((dimension, dimension), 0.07)
        self.inner_window = 
        self.frame_left = round(dimension * 0.15)
        self.frame_top = round(dimension * 0.27)
        self.frame_right = round(dimension * 0.85)
        self.frame_bottom = round(dimension * 0.73)
        self.large_dim = round(dimension * 0.688)
        self.small_dim = round(dimension * 0.459)
        
    template = slide(tmp_w)"""

def greasepencil(input_image, ncol=4, dimension=None):

    #os.chdir(f"{source}")

    # read image and determine template width
    color_image = io.imread(input_image)
    image = img_as_ubyte(io.imread(input_image, as_gray=True))
    m, n = image.shape
    if dimension is None:
        tmp_w = round(min(image.shape)/ncol)
    else:
        tmp_w = dimension
    # establish relative measures
    frame_left = round(tmp_w * 0.15)
    frame_right = round(tmp_w * 0.85)
    frame_top = round(tmp_w * 0.27)
    frame_bottom = round(tmp_w * 0.73)

    # build slide template
    rel_template = np.full((tmp_w, tmp_w), 0.07)
    rel_template[frame_top:frame_bottom, frame_left:frame_right] = 0.5
    rel_template = np.pad(rel_template, (30,), constant_values=1)
    rel_template += 0.1 * np.random.random(rel_template.shape)

    # loop over template sizes and pick best match
    widths = []
    coors = []
    h_results = []
    v_results = []
    found = None

    for scale in np.linspace(0.5, 1.0, 6)[::-1]:
        
        tmp_w = int(rel_template.shape[0]*scale)
        resized = transform.resize(rel_template, (tmp_w, tmp_w))

        v_template = np.rot90(resized)

        h_result = match_template(color_image[:,:,2], resized, pad_input=True)
        v_result = match_template(color_image[:,:,2], v_template, pad_input=True)

        h_coordinates = peak_local_max(h_result, min_distance=int(tmp_w*0.8), threshold_abs=0.35, exclude_border=False)
        coordinates_list = [[coor[0], coor[1]] for coor in h_coordinates]

        if dimension:
            break

        peaks = [h_result[r, c] for r, c in h_coordinates]
        
        if peaks:
            avg = sum(peaks)/len(peaks)
        else:
            continue
        
        if found is None or avg > found:
            found = avg
            widths.append(tmp_w)
            coors.append(coordinates_list)
            h_results.append(h_result)
            v_results.append(v_result)
        else:
            coordinates_list =  coors[-1]
            v_result = v_results[-1]
            h_result = h_results[-1]
            tmp_w = widths[-1]
            break



    fig = plt.figure(figsize=(8, 3))
    gs = fig.add_gridspec(2, 4)
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[1, 0])
    ax3 = plt.subplot(gs[:, 1])
    ax4 = plt.subplot(gs[:, 2], sharex=ax3, sharey=ax3)
    ax5 = plt.subplot(gs[:, 3], sharex=ax3, sharey=ax3)

    ax1.imshow(rel_template)
    ax1.set_axis_off()
    ax1.set_title('horizontal template')


    ax2.imshow(v_template)
    ax2.set_axis_off()
    ax2.set_title('vertical template')


    ax3.imshow(color_image)
    ax3.set_axis_off()
    ax3.set_title(f'detected frames (tmp_w:{tmp_w})')

    for point in coordinates_list:
        r, c = point
        slide = mpatches.Rectangle((c - tmp_w/2, r - tmp_w/2), tmp_w, tmp_w,
                            fill=False, edgecolor='red', linewidth=1) 
        if v_result[r, c] > h_result[r, c]:
            picture = mpatches.Rectangle((c - round(tmp_w * 0.23), r - round(tmp_w * 0.35)), round(tmp_w * 0.23)*2, round(tmp_w * 0.35)*2,
                            fill=False, edgecolor='red', linewidth=1)   
        else:
            picture = mpatches.Rectangle((c - round(tmp_w * 0.35), r - round(tmp_w * 0.23)), round(tmp_w * 0.35)*2, round(tmp_w * 0.23)*2, 
                            fill=False, edgecolor='red', linewidth=1)
        ax3.add_patch(slide)
        ax3.add_patch(picture)


    ax4.imshow(h_result)
    #ax4.plot(h_coordinates[:,1], h_coordinates[:,0], 'r.')
    ax4.set_axis_off()
    ax4.set_title('horizontal matching')


    ax5.imshow(v_result)
    #ax5.plot(v_coordinates[:,1], v_coordinates[:,0], 'r.')
    ax5.set_axis_off()
    ax5.set_title('vertical matching')

    plt.show()

    
    """
    counter = 1

    for point in (coordinates_list):
        r, c = point
        name, ext = str.split(input_image, ".")
        if v_result[r, c] > h_result[r, c]:
            io.imsave(
                f"{destination}/{name}-{counter:02}.jpg", 
                color_image
                [r - round(tmp_w * 0.35) : r + round(tmp_w * 0.35), 
                c - round(tmp_w * 0.23) : c + round(tmp_w * 0.23)], 
                quality=100
            )
        else:       
            io.imsave(
                f"{destination}/{name}-{counter:02}.jpg",
                color_image
                [r - round(tmp_w * 0.23) : r + round(tmp_w * 0.23), 
                c - round(tmp_w * 0.35) : c + round(tmp_w * 0.35)], 
                quality=100
            )
        
        counter += 1



extensions = (".jpg", ".jpeg")
image_files = [
    image_file
    for image_file in os.listdir(f"{source}")
    if str(image_file).endswith(tuple(extensions))
]

# call function
#for item in image_files:
    try:
        greasepencil(item)
    except IndexError:
        print(f"error: {i}")

#final feedback
#print(f"Você forneceu {len(image_files)} imagens e gerou {len(os.listdir(destination))} arquivos!")"""

greasepencil("/Users/martimpassos/dev/greasepencil/_errors/P012V001-0385.jpg")