# import json
# import os
# import cv2
# import numpy as np
# from tqdm import tqdm
# from pycocotools import mask as maskUtils

# parent_path = '/data/zequn/datasets/coco/val2017'
# json_file = '/data/zequn/datasets/coco/annotations/instances_val2017.json'
# with open(json_file) as anno_:
#     annotations = json.load(anno_)
# # import pdb; pdb.set_trace()
# def apply_mask(image, segmentation):
#     alpha = 0.5
#     color = (0, 0.6, 0.6)
#     threshold = 0.5
#     mask = maskUtils.decode(segmentation) # 分割解码
#     mask = np.where(mask >= threshold, 1, 0).astype(np.uint8)
#     for c in range(3): # 3个通道
#         # mask=1执行前一个，否则后一个
#         image[:, :, c] = np.where(mask == 1,
#                                   image[:, :, c] *
#                                   (1 - alpha) + alpha * color[c] * 255,
#                                   image[:, :, c])
#     return image

# results = []
# for i in range(len(annotations['annotations'])):
#     image_id = annotations['annotations'][i]['image_id']
#     # 包含size:图片高度宽度  counts:压缩后的mask  通过mask = maskUtils.decode(encoded_mask)解码，得到mask,需要导入from pycocotools import mask as maskUtils
#     segmentation = annotations['annotations'][i]['segmentation'] 
#     full_path = os.path.join(parent_path, str(image_id).zfill(12) + '.jpg')
#     image = cv2.imread(full_path)
#     mask_image = apply_mask(image, segmentation)
#     cv2.imshow('demo', mask_image)
#     cv2.waitKey(5000)
    
    



from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import os
pylab.rcParams['figure.figsize'] = (8.0, 10.0)
 
dataDir = '/data/zequn/datasets/coco'
dataType = 'val2017'
annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
 
# I为图片具体位置

imgs = os.listdir('/data/zequn/datasets/coco/val2017')
for i, img in enumerate(imgs):
    if i < 100:
        continue
    if i == 200:
        break
    I = io.imread(dataDir + '/val2017/' + img)
    
    coco = COCO(annFile)
    
    plt.imshow(I)
    plt.axis('off')
    annIds = coco.getAnnIds(imgIds=int(img.split('.')[0]), iscrowd=None)
    # print(annIds)
    anns = coco.loadAnns(annIds)
    # print(anns)
    coco.showAnns(anns)
    plt.savefig('vis_coco/{}.png'.format(i+1))
    plt.clf()