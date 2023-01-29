import numpy as np
import os 
from tqdm import tqdm 
# from utils import levelset2boundary, levelset2boundary2D
from torchvision.transforms.functional import rgb_to_grayscale
import skimage
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import cv2 


def cropROI(image, label):
    fg = (label > 0).astype(np.uint8) * 255
    cnt, _ = cv2.findContours(fg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rect = cv2.minAreaRect(cnt[0])
    pts2 = cv2.boxPoints(rect)
    H, W = rect[1]
    H, W = int(H), int(W)
    pts1 = np.float32([[0,0],[W,0],[W, H],[0,H]])
    M = cv2.getPerspectiveTransform(pts2,pts1)
    label_out = cv2.warpPerspective(label,M,(W, H), flags=cv2.INTER_NEAREST)
    label_out = mergeClass(label_out)
    img_out = cv2.warpPerspective(image, M,(W, H))

    return img_out, label_out

def mergeClass(label):
    # assert 0 not in np.unique(label)
    cls = np.unique(label)
    for i in cls:
        if i not in [1,2,3,4]:
            if i in [19, 20]:
                label[label == i] = 1 
            elif i in [14, 10, 11]:
                label[label == i] = 3 
            else:
                label[label == i] = 0

    return label

def splitImg(img, label, tile=1024, stride=512):
    tile_imgs = []
    tile_labels = []
    H, W = label.shape

    x1 = 0
    x2 = tile
    while x1 < W:
        y1 = 0
        y2 = tile
        while y1 < H:
            tile_img = img[y1:y2,x1:x2]
            tile_label = label[y1:y2,x1:x2]
            tile_imgs.append(tile_img)
            tile_labels.append(tile_label)
            # print('y:{:}-{:}'.format(y1,y2), 'x:{:}-{:}'.format(x1,x2))

            y1 += stride
            y2 += stride

            if y2 >= H:
                y1, y2 = H-tile, H
                tile_img = img[y1:y2,x1:x2]
                tile_label = label[y1:y2,x1:x2]
                tile_imgs.append(tile_img)
                tile_labels.append(tile_label)
                # print('y:{:}-{:}'.format(y1,y2), 'x:{:}-{:}'.format(x1,x2))
                break

        x1 += stride
        x2 += stride
        
        if x2 >= W:
            y1 = 0
            y2 = tile
            x1, x2 = W - tile, W
            while y1 < H:
                # print('y:{:}-{:}'.format(y1,y2), 'x:{:}-{:}'.format(x1,x2))
                tile_img = img[y1:y2,x1:x2]
                tile_label = label[y1:y2,x1:x2]
                tile_imgs.append(tile_img)
                tile_labels.append(tile_label)

                y1 += stride
                y2 += stride

                if y2 >= H:
                    y1, y2 = H-tile, H
                    tile_img = img[y1:y2,x1:x2]
                    tile_label = label[y1:y2,x1:x2]
                    tile_imgs.append(tile_img)
                    tile_labels.append(tile_label)
                    # print('y:{:}-{:}'.format(y1,y2), 'x:{:}-{:}'.format(x1,x2))
                    break
            break 
    
    return tile_imgs, tile_labels


if __name__ == '__main__':
    origin_dir = '../dataset2d/BCSS/'
    out_dir = '/dataset/BCSS/images/bcss/'
    imgs_dir = os.path.join(origin_dir, 'images/')
    segs_dir = os.path.join(origin_dir, 'masks/')
    imgs = []
    segs = []

    nms = sorted(['.'.join(nm.split('.')[:-1]) for nm in os.listdir(imgs_dir)])

    for nm in nms:
        img_path = os.path.join(imgs_dir, nm + '.png')
        seg_path = os.path.join(segs_dir, nm + '.png')

        if os.path.isfile(img_path) & os.path.isfile(seg_path):
            imgs.append(img_path)
            segs.append(seg_path)

    files = [{"image": img, "label": seg} for img, seg in zip(imgs, segs)]

    for idx, sample in tqdm(enumerate(files), total=len(files)):
        img_path = sample['image']
        label_path = sample['label']
        img = Image.open(img_path)
        label = Image.open(label_path)

        img = np.array(img)
        label = np.array(label)

        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)
            print('gray image : ', img_path)
        
        if (len(img.shape) == 3) & (img.shape[-1] == 4):
            img = skimage.color.rgba2rgb(img)
            img = (img * 255).astype(np.uint8)
            print("rgba image : ", img_path)


        if 0 in np.unique(label):
            img, label = cropROI(img, label)
            img = img[5:-5,5:-5]
            label = label[5:-5,5:-5]
        else:
            label = mergeClass(label)
        
        img_outpath = '/dataset/BCSS/images_process/bcss' + str(idx).zfill(5) + '.png'
        label_outpath = '/dataset/BCSS/masks_process/bcss' + str(idx).zfill(5) + '.png'
        
        img_out = Image.fromarray(img)
        img_out.save(img_outpath)

        label_out = Image.fromarray(label)
        label_out.save(label_outpath)    

    # split processed images to patches
    data_dir = '/dataset/BCSS/'
    tileimgs_dir = os.path.join(data_dir, 'images/')
    tilesegs_dir = os.path.join(data_dir, 'masks/')
    
    imgs_dir = os.path.join(data_dir, 'images_process/')
    segs_dir = os.path.join(data_dir, 'masks_process/')
    imgs = []
    segs = []

    nms = sorted(['.'.join(nm.split('.')[:-1]) for nm in os.listdir(imgs_dir)])

    for nm in nms:
        img_path = os.path.join(imgs_dir, nm + '.png')
        seg_path = os.path.join(segs_dir, nm + '.png')

        if os.path.isfile(img_path) & os.path.isfile(seg_path):
            imgs.append(img_path)
            segs.append(seg_path)

    files = [{"image": img, "label": seg} for img, seg in zip(imgs, segs)]

    for idx, sample in tqdm(enumerate(files), total = len(files)):
        img_path = sample['image']
        label_path = sample['label']
        img = Image.open(img_path)
        label = Image.open(label_path)

        img = np.array(img)
        label = np.array(label)

        nm_base = img_path.split('/')[-1].split('.')[0]
        tile_imgs, tile_labels = splitImg(img, label, tile=1024, stride=512)
        tile_id = 0
        for tile_img, tile_label in zip(tile_imgs, tile_labels):
            out_nm = nm_base + '-' + str(tile_id).zfill(2) + '.png'

            img_outpath = os.path.join(tileimgs_dir, out_nm)
            tile_img_out = Image.fromarray(tile_img)
            tile_img_out.save(img_outpath)
 
            label_outpath = os.path.join(tilesegs_dir, out_nm)
            tile_label_out = Image.fromarray(tile_label)
            tile_label_out.save(label_outpath)

            tile_id += 1
