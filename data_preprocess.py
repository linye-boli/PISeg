import numpy as np
import os 
import glob 
from skimage import segmentation
import monai.transforms as transforms
from monai.data import CacheDataset, DataLoader, Dataset
import skfmm
from tqdm import tqdm 
from utils import levelset2boundary, levelset2boundary2D
from torchvision.transforms.functional import rgb_to_grayscale
import torch 

def preprocess_3d_dataset(
    origin_dir, 
    out_dir, 
    margin=(16, 16, 48),       # margin to expand
    spatial_size=(128,128,48), # size to pad
    num_workers=16):
    
    # load origin
    images = sorted(
        glob.glob(os.path.join(origin_dir, "imagesTr", "*.nii.gz")))
    labels = sorted(
        glob.glob(os.path.join(origin_dir, "labelsTr", "*.nii.gz")))

    data_dicts = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(images, labels)
    ]

    basic_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=['image', 'label']),
            transforms.EnsureChannelFirstd(keys=["image", "label"]),
            transforms.Orientationd(
                keys=["image", "label"], 
                axcodes="RAS"),
            transforms.CropForegroundd(
                keys=["image", "label"], 
                source_key="label",
                margin=margin),
            transforms.Resized(
                keys=['image', 'label'],
                spatial_size=spatial_size,
                mode=['trilinear','nearest'])])

    ds = CacheDataset(
        data=data_dicts, transform=basic_transforms,
        cache_rate=1.0, num_workers=num_workers)
    
    # create dataset file structure
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    out_imgdir = os.path.join(out_dir, 'imagesTr')
    out_labeldir = os.path.join(out_dir, 'labelsTr')
    out_sdfdir = os.path.join(out_dir, 'sdfsTr')
    out_boundarydir = os.path.join(out_dir, 'boundariesTr')
    
    
    if not os.path.exists(out_imgdir):
        os.makedirs(out_imgdir)
    
    if not os.path.exists(out_labeldir):
        os.makedirs(out_labeldir)
    
    if not os.path.exists(out_sdfdir):
        os.makedirs(out_sdfdir)
    
    if not os.path.exists(out_boundarydir):
        os.makedirs(out_boundarydir)
    
    for sample in tqdm(ds, total=len(ds)):
        fnm = sample['image_meta_dict']['filename_or_obj'].split('/')[-1].split('.')[0]

        img_outpath = os.path.join(out_imgdir, fnm + '.npz')
        label_outpath = os.path.join(out_labeldir, fnm + '.npz')
        sdf_outpath = os.path.join(out_sdfdir, fnm+'.npz')
        boundary_outpath = os.path.join(out_boundarydir, fnm+'.npz')

        if not os.path.exists(img_outpath):
            image = sample['image']
            image = image.numpy().astype('int16')
            np.savez_compressed(img_outpath, image)
            print('save image at : ', img_outpath)
        
        if not os.path.exists(label_outpath):
            label = sample['label']
            label = label.numpy().astype('uint8')
            np.savez_compressed(label_outpath, label)
            print('save label at : ', label_outpath)

        # if not os.path.exists(sdf_outpath):
        #     label = sample['label']
        #     label = label[0].numpy().astype('uint8')
        #     sdf = []
        #     boundary = []
        #     cls = np.unique(label)
        #     for c in cls:
        #         sdf_ = skfmm.distance(0.5 - (label==c))
        #         bd_ = levelset2boundary(sdf_)
        #         sdf_ = sdf_ - sdf_ * bd_
        #         sdf.append(sdf_)
        #         boundary.append(bd_)
            
        #     sdf = np.stack(sdf, axis=0).astype(np.float32)
        #     np.savez_compressed(sdf_outpath, sdf)
        #     print('save sdf at : ', sdf_outpath)

        #     boundary = np.stack(boundary, axis=0).astype(np.bool)
        #     np.savez_compressed(boundary_outpath, boundary)
        #     print('save boundary at : ', boundary_outpath)
        
        if not os.path.exists(sdf_outpath):
            label = sample['label']
            label = label[0].numpy().astype('uint8')
            sdf = []
            boundary = []
            cnt = []

            cls = np.unique(label)
            for c in cls:
                # inside object is positive
                label_ = label == c
                label_ = label_ - 0.5
                bd_ = levelset2boundary(label_)
                ls_ = label_*(1-bd_)
                sdf_ = skfmm.distance(ls_, dx=(1/(spatial_size[0]-1), 1/(spatial_size[1]-1), 1/(spatial_size[2]-1)), order=1)
                sdf.append(sdf_)
                boundary.append(bd_)
            
            sdf = np.stack(sdf, axis=0).astype(np.float32)
            np.savez_compressed(sdf_outpath, sdf)
            print('save sdf at : ', sdf_outpath)

            boundary = np.stack(boundary, axis=0).astype(np.bool)
            np.savez_compressed(boundary_outpath, boundary)
            print('save boundary at : ', boundary_outpath)

import pdb 
from utils import resize
def preprocess_2d_dataset(
    origin_dir, 
    out_dir, 
    margin=256,       # margin to expand
    spatial_size=(64, 128), # size to pad
    num_workers=16,
    num_cls = 2,
    cache_rate=1,
    img_postfix='.png', 
    seg_postfix='_mask.png'):
    
    imgs_dir = os.path.join(origin_dir, 'images')
    segs_dir = os.path.join(origin_dir, 'masks')

    imgs = []
    segs = []
    nms = [nm.split('.')[0] for nm in os.listdir(imgs_dir)]
    for nm in nms:
        img_path = os.path.join(imgs_dir, nm + img_postfix)
        seg_path = os.path.join(segs_dir, nm + seg_postfix)

        if os.path.isfile(img_path) & os.path.isfile(seg_path):
            imgs.append(img_path)
            segs.append(seg_path)

    files = [{"image": img, "label": seg} for img, seg in zip(imgs, segs)]

    trans =  transforms.Compose(
        [
            transforms.LoadImaged(keys=['image', 'label']),
            transforms.EnsureChannelFirstd(keys=["image", "label"]),
            transforms.CropForegroundd(keys=['image', 'label'], source_key='label', margin=margin),
            # transforms.Resized(keys=['image', 'label'], spatial_size=spatial_size, mode=['bilinear', 'nearest'])
            ])

    ds = CacheDataset(
        data=files, transform=trans,
        cache_rate=cache_rate, num_workers=num_workers)
    
    # create dataset file structure
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    out_imgdir = os.path.join(out_dir, 'imagesTr')
    out_labeldir = os.path.join(out_dir, 'labelsTr')
    out_sdfdir = os.path.join(out_dir, 'sdfsTr')
    out_boundarydir = os.path.join(out_dir, 'boundariesTr')
    out_cntdir = os.path.join(out_dir, 'cntsTr')
    
    if not os.path.exists(out_imgdir):
        os.makedirs(out_imgdir)
    
    if not os.path.exists(out_labeldir):
        os.makedirs(out_labeldir)
    
    if not os.path.exists(out_sdfdir):
        os.makedirs(out_sdfdir)
    
    if not os.path.exists(out_boundarydir):
        os.makedirs(out_boundarydir)
    
    if not os.path.exists(out_cntdir):
        os.makedirs(out_cntdir)
    
    for sample in tqdm(ds, total=len(ds)):
        fnm = sample['image_meta_dict']['filename_or_obj'].split('/')[-1].split('.')[0]

        img_outpath = os.path.join(out_imgdir, fnm + '.npz')
        label_outpath = os.path.join(out_labeldir, fnm + '.npz')
        sdf_outpath = os.path.join(out_sdfdir, fnm+'.npz')
        boundary_outpath = os.path.join(out_boundarydir, fnm+'.npz')
        cnt_outpath = os.path.join(out_cntdir, fnm+'.npz')

        if not os.path.exists(img_outpath):
            image = sample['image']
            if image.shape[0] != 1:
                image = rgb_to_grayscale(image)
            image = resize(image[None], spatial_size)
            image = image.numpy().astype('uint8')[0]
            np.savez_compressed(img_outpath, image)
            print('save image at : ', img_outpath)
        
        if not os.path.exists(label_outpath):
            label = sample['label']
            label = resize(label[None], spatial_size, mode='nearest')[0]
            label = label.numpy().astype('uint8')
            np.savez_compressed(label_outpath, label)
            print('save label at : ', label_outpath)

        if (not os.path.exists(sdf_outpath)) or (not os.path.exists(cnt_outpath)):
            label = sample['label']
            label = label[0].numpy().astype('uint8')
            nx, ny = label.shape
            sdf = []
            boundary = []
            cnt = []

            cls = np.arange(num_cls)
            for c in cls:

                # inside object is positive
                label_ = label == c
                label_ = label_ - 0.5

                try:
                    bd_ = levelset2boundary2D(label_, False) # boundary problem for complex image
                    sdf_ = skfmm.distance(label_, dx=(1/nx, 1/ny))
                except:
                    sdf_ = torch.zeros_like(torch.tensor(label_))
                    bd_ = torch.ones_like(torch.tensor(label_))

                sdf_ = torch.tensor(sdf_)[None][None]
                bd_ = torch.tensor(bd_)[None][None]
                
                sdf_ = resize(sdf_, spatial_size)[0][0].numpy()
                bd_ =  resize(bd_, spatial_size)[0][0].numpy()
                sdf.append(sdf_)
                boundary.append(bd_ > 0.5)
            
            sdf = np.stack(sdf, axis=0).astype(np.float32)
            np.savez_compressed(sdf_outpath, sdf)
            print('save sdf at : ', sdf_outpath)

            boundary = np.stack(boundary, axis=0).astype(np.bool_)
            np.savez_compressed(boundary_outpath, boundary)
            print('save boundary at : ', boundary_outpath)

            # cnt = np.stack(cnt, axis=0).astype(np.float32)
            # np.savez_compressed(cnt_outpath, cnt)
            # print('save cnt at : ', cnt_outpath)


if __name__ == '__main__':

    # preprocess MSD Spleen dataset
    # preprocess_3d_dataset(
    #     origin_dir='/dataset/MSD/Task09_Spleen/', 
    #     out_dir='/dataset/MSD/SpleenPreprocess_96x96x32/', 
    #     margin=(32, 32, 32), 
    #     spatial_size=(96,96,32),
    #     num_workers=16)
    
    # preprocess_3d_dataset(
    #     origin_dir='/dataset/MSD/Task09_Spleen/', 
    #     out_dir='/dataset/MSD/SpleenPreprocess_192x192x64/', 
    #     margin=(32, 32, 32), 
    #     spatial_size=(192,192,64),
    #     num_workers=16)    
    
    # # preprocess segthor dataset
    # preprocess_3d_dataset(
    #     '/dataset/segthor/segthor/', 
    #     '/dataset/segthor/segthorPreprocess/', 
    #     margin=(64, 64, 64), 
    #     spatial_size=(128,128,128),
    #     num_workers=16)
    
    # preprocess CXR SHENZHEN leftlung dataset 
    # preprocess_2d_dataset(
    #     origin_dir = '/workdir/PISegFull/dataset2d',
    #     out_dir = '/dataset/CXR/leftlungSZPreprocess_128x128',
    #     margin=256,
    #     spatial_size=(128, 128)
    # )

    # preprocess_2d_dataset(
    #     origin_dir = '/dataset/CXR',
    #     out_dir = '/dataset/CXR/leftlungSZPreprocess_64x64',
    #     margin=256,
    #     spatial_size=(64, 64)
    # )

    # preprocess_2d_dataset(
    #     origin_dir = '/dataset/CXR',
    #     out_dir = '/dataset/CXR/leftlungSZPreprocess_128x128',
    #     margin=256,
    #     spatial_size=(128, 128)
    # )
    
    # preprocess_2d_dataset(
    #     origin_dir = '/dataset/CXR',
    #     out_dir = '/dataset/CXR/leftlungSZPreprocess_256x256',
    #     margin=256,
    #     spatial_size=(256, 256)
    # )

    # preprocess_2d_dataset(
    #     origin_dir = '/dataset/CXR',
    #     out_dir = '/dataset/CXR/leftlungSZPreprocess_512x512',
    #     margin=256,
    #     spatial_size=(512, 512)
    # )

    # preprocess_2d_dataset(
    #     origin_dir = '/dataset/Oxford-III_PET',
    #     out_dir = '/dataset/Oxford-III_PET/PETPreprocess_128x128',
    #     margin=64,
    #     spatial_size=(128, 128),
    #     img_postfix='.jpg',
    #     seg_postfix='.png',
    # )

    # preprocess_2d_dataset(
    #     origin_dir = '/dataset/Oxford-III_PET',
    #     out_dir = '/dataset/Oxford-III_PET/PETPreprocess_256x256',
    #     margin=64,
    #     spatial_size=(256, 256),
    #     img_postfix='.jpg',
    #     seg_postfix='.png',
    # )

    # preprocess_2d_dataset(
    #     origin_dir = '/dataset/Oxford-III_PET',
    #     out_dir = '/dataset/Oxford-III_PET/PETPreprocess_512x512',
    #     margin=64,
    #     spatial_size=(512, 512),
    #     img_postfix='.jpg',
    #     seg_postfix='.png',
    # )
    
    # preprocess_2d_dataset(
    #     origin_dir = '/dataset/CRAG',
    #     out_dir = '/dataset/CRAG/CRAGPreprocess_256x256',
    #     margin=0,
    #     num_cls=2,
    #     spatial_size=(256, 256),
    #     img_postfix='.png',
    #     seg_postfix='.png',
    # )

    preprocess_2d_dataset(
        origin_dir = '/dataset/CRAG',
        out_dir = '/dataset/CRAG/CRAGPreprocess_512x512',
        margin=0,
        num_cls=2,
        spatial_size=(512, 512),
        img_postfix='.png',
        seg_postfix='.png',
    )

    preprocess_2d_dataset(
        origin_dir = '/dataset/CRAG',
        out_dir = '/dataset/CRAG/CRAGPreprocess_800x800',
        margin=0,
        num_cls=2,
        spatial_size=(800, 800),
        img_postfix='.png',
        seg_postfix='.png',
    )