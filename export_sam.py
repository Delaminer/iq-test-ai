import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import distance as dist
import math
from scipy import ndimage
from skimage import measure
import numpy as np
import torch
from raven_dataset import dataset
from torchvision import transforms, utils
from math import ceil
from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry

path = "/nfs/turbo/coe-chaijy-unreplicated/datasets/RAVEN-10000"
train = dataset(path, "train", 224)

sam = sam_model_registry["vit_h"](checkpoint="/nfs/turbo/coe-chaijy-unreplicated/datasets/SAM/sam_vit_h_4b8939.pth")
predictor = SamPredictor(sam)
mask_generator = SamAutomaticMaskGenerator(sam, points_per_side=32, box_nms_thresh=0.7)

def novelty(i, masks,debug=False):
    mask = masks[i]
    # how much has this mask added, compared to masks before it?
    # keep removing other masks from this, and return the final sum
    novel_mask = mask['segmentation']
    if debug:
        print("Before")
        plt.imshow(novel_mask)
        plt.show()
    for before in range(i):
        novel_mask = np.bitwise_and(novel_mask, np.bitwise_not(masks[before]['segmentation']))
        novel_mask = np.bitwise_and(novel_mask, np.bitwise_not(masks[before]['segmentation']))
    if debug:
        plt.show("After")
        plt.imshow(novel_mask)
        plt.show()
    return novel_mask.sum() / mask['segmentation'].sum()

def chunk_up(masks, image):
    # Chunk up the image into its 4x4 cells.
    # Put each mask in which cell it belongs to
    cells = [[[] for _ in range(4)] for _ in range(4)]
    cell_width = image.shape[1] // 4
    cell_height = image.shape[0] // 4
    for i, mask in enumerate(masks):
        x, y, w, h = mask['bbox']
        x1, y1, x2, y2 = x, y, x + w, y + h
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)
        x1_id = x1 // cell_width
        y1_id = y1 // cell_height
        x2_id = x2 // cell_width
        y2_id = y2 // cell_height
        # print(f"i={(i)}, x1_id={x1_id}, y1_id={y1_id}, x2_id={x2_id}, y2_id={y2_id} from {mask['bbox']}")
        if x1_id == x2_id and y1_id == y2_id:
            cells[y1_id][x1_id].append(mask)
            
    # flatten the 2d array
    return [cell for row in cells for cell in row]

# save this cells data to cells.npz
mask_dtype = np.dtype([('bbox', 'intc', (4,)), ('segmentation', object), ('area', float), ('predicted_iou', float)])
fields = ['bbox', 'segmentation', 'area', 'predicted_iou']

def export_question(qid):
    images, target, meta_target, meta_structure, embedding, indicator = train.__getitem__(qid)
    base_image_shape = images[0].shape
    # Create one large 4x4 image
    image = np.zeros((base_image_shape[0] * 4, base_image_shape[1] * 4))
    for i in range(4):
        for j in range(4):
            image[i * base_image_shape[0]: (i + 1) * base_image_shape[0], j * base_image_shape[1]: (j + 1) * base_image_shape[1]] = images[i * 4 + j]
    image = image.reshape(image.shape[0], image.shape[1], 1)
    image = np.concatenate([image, image, image], axis=2)
    masks = mask_generator.generate(image)
    masks.sort(key=(lambda x: x['predicted_iou']), reverse=True)
    cells = chunk_up(masks, image)

    structured_cells = []
    obj = dict([(field, []) for field in fields])
    sizes = []
    for cell in cells:
        # show_anns(image, cell)
        structured_cell = np.zeros(len(cell), dtype=mask_dtype)
        for i, mask in enumerate(cell):
            for field in fields:
                structured_cell[i][field] = mask[field]
                obj[field].append(mask[field])
        structured_cells.append(structured_cell)
        sizes.append(len(cell))
    np.savez(f'sam_output/sam_cell_{qid}.npz', cell_sizes=sizes, **obj)

start = 30100
end = start + 100
for i in range(start, end):
    export_question(i)
    print('Exported ', i)
    # if i % 100 == 0:
    #     print("Exported ", i)
print("DONE")