import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.feature import match_template, peak_local_max
from skimage.morphology import extrema
from skimage.measure import label
from skimage.segmentation import watershed
from skimage import exposure, transform, img_as_ubyte, io, exposure, color, feature

    
def greasepencil(input_image, ncol=4, tmp_w=None):

    color_image = io.imread(input_image)
    image = img_as_ubyte(io.imread(input_image, as_gray=True))

    if tmp_w is None:
        tmp_w = round(min(image.shape)/ncol)


    frame_left = round(tmp_w * 0.15)
    frame_right = round(tmp_w * 0.85)
    frame_top = round(tmp_w * 0.27)
    frame_bottom = round(tmp_w * 0.73)


    rel_template = np.full((tmp_w, tmp_w), 0.07)
    rel_template[frame_top, frame_left:frame_right] = 0.5
    rel_template[frame_bottom, frame_left:frame_right] = 0.5
    rel_template[frame_top:frame_bottom, frame_left] = 0.5
    rel_template[frame_top:frame_bottom, frame_right] = 0.5
    rel_template = np.pad(rel_template, (20,), constant_values=1)
    rel_template += 0.1 * np.random.random(rel_template.shape)

    template_edges = feature.canny(rel_template)
    image_edges = feature.canny(color_image[:,:,2])

    h_result = match_template(image_edges, template_edges, pad_input=True)
    h_coordinates = peak_local_max(h_result, min_distance=int(tmp_w*0.8), threshold_abs=0.35, exclude_border=False)

    fig = plt.figure(figsize=(8, 3))
    gs = fig.add_gridspec(2, 4)
    ax1 = plt.subplot(gs[0, 0])
    #ax2 = plt.subplot(gs[1, 0])
    ax3 = plt.subplot(gs[:, 1])
    ax4 = plt.subplot(gs[:, 2], sharex=ax3, sharey=ax3)
    #ax5 = plt.subplot(gs[:, 3], sharex=ax3, sharey=ax3)

    ax1.imshow(template_edges)
    ax1.set_axis_off()
    ax1.set_title('horizontal template')


    #ax2.imshow(v_template)
    #ax2.set_axis_off()
    #ax2.set_title('vertical template')


    ax3.imshow(image_edges)
    ax3.set_axis_off()
    ax3.set_title(f'detected frames (tmp_w:{tmp_w})')

    for point in h_coordinates:
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


    #ax5.imshow(v_result)
    #ax5.plot(v_coordinates[:,1], v_coordinates[:,0], 'r.')
    #ax5.set_axis_off()
    #ax5.set_title('vertical matching')

    plt.show()

greasepencil("/Users/martimpassos/dev/greasepencil/_input/01.jpg", tmp_w=930)