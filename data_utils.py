import numpy as np
import os 
import glob 
from skimage import segmentation
import monai.transforms as transforms
from monai.data import CacheDataset, DataLoader, Dataset
import skfmm
from tqdm import tqdm 

def get_unet3d_loader(
    traindata_dir, 
    train_batchsize=2, 
    num_workers=4,
    num_train=31, 
    a_min=-57, # intensity lower bound
    a_max=164, # intensity upper bound
    sample=False):

    train_images = sorted(
        glob.glob(os.path.join(traindata_dir, "imagesTr", "*.npz")))
    train_labels = sorted(
        glob.glob(os.path.join(traindata_dir, "labelsTr", "*.npz")))
    train_sdfs = sorted(
        glob.glob(os.path.join(traindata_dir, "sdfsTr", "*.npz")))
    train_boundaries = sorted(
        glob.glob(os.path.join(traindata_dir, "boundariesTr", "*.npz")))

    data_dicts = [
        {"image": image_name, "label": label_name, 'sdf':sdf_name, "boundary": boundary_name}
        for image_name, label_name, sdf_name, boundary_name in zip(train_images, train_labels, train_sdfs, train_boundaries)
    ]

    if sample:
        train_files, val_files = data_dicts[:4], data_dicts[-4:]
    else:
        train_files, val_files = data_dicts[:num_train], data_dicts[num_train:]

        # setup transforms for training and validation 
            # setup transforms for training and validation 
    train_transforms = transforms.Compose(
        [
            transforms.LoadImaged(
                keys=["image", "label", "sdf", "boundary"], channel_dim=0),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=a_min, a_max=a_max,
                b_min=0.0, b_max=1.0, clip=True),
            transforms.RandRotated(
                keys=["image", "label", "sdf", "boundary"], 
                range_x=0.4, range_y=0.4, range_z=0.4, 
                prob=0.1, keep_size=True, mode=['bilinear', 'nearest', 'bilinear', 'nearest'],  align_corners=True)
        ])

    val_transforms = transforms.Compose(
        [
            transforms.LoadImaged(
                keys=["image", "label", "sdf", "boundary"], channel_dim=0),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=a_min, a_max=a_max,
                b_min=0.0, b_max=1.0, clip=True),
        ])

    
    # define cachedataset and dataloader for training and validation
    train_ds = CacheDataset(
        data=train_files, transform=train_transforms,
        cache_rate=1.0, num_workers=num_workers)
    train_loader = DataLoader(
        train_ds, batch_size=train_batchsize, 
        shuffle=True, num_workers=num_workers)
    
    val_ds = CacheDataset(
        data=val_files, transform=val_transforms, 
        cache_rate=1.0, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=num_workers)

    return train_loader, val_loader

def get_unet2d_loader(
    traindata_dir, 
    valdata_dir,
    train_batchsize=2, 
    num_workers=4,
    num_train=500, 
    a_min=0, # intensity lower bound
    a_max=255, # intensity upper bound
    sample=False):

    train_images = sorted(
        glob.glob(os.path.join(traindata_dir, "imagesTr", "*.npz")))
    train_labels = sorted(
        glob.glob(os.path.join(traindata_dir, "labelsTr", "*.npz")))
    train_sdfs = sorted(
        glob.glob(os.path.join(traindata_dir, "sdfsTr", "*.npz")))
    train_boundaries = sorted(
        glob.glob(os.path.join(traindata_dir, "boundariesTr", "*.npz")))
    
    val_images = sorted(
        glob.glob(os.path.join(valdata_dir, "imagesTr", "*.npz")))
    val_labels = sorted(
        glob.glob(os.path.join(valdata_dir, "labelsTr", "*.npz")))
    val_sdfs = sorted(
        glob.glob(os.path.join(valdata_dir, "sdfsTr", "*.npz")))
    val_boundaries = sorted(
        glob.glob(os.path.join(valdata_dir, "boundariesTr", "*.npz")))

    traindata_dicts = [
        {"image": image_name, "label": label_name, 'sdf':sdf_name, "boundary": boundary_name}
        for image_name, label_name, sdf_name, boundary_name in zip(train_images, train_labels, train_sdfs, train_boundaries)]
    
    valdata_dicts = [
        {"image": image_name, "label": label_name, 'sdf':sdf_name, "boundary": boundary_name}
        for image_name, label_name, sdf_name, boundary_name in zip(val_images, val_labels, val_sdfs, val_boundaries)]

    if sample:
        train_files, val_files = traindata_dicts[:40], valdata_dicts[-40:]
    else:
        train_files, val_files = traindata_dicts[:num_train], valdata_dicts[num_train:]

        # setup transforms for training and validation 
            # setup transforms for training and validation 
    train_transforms = transforms.Compose(
        [
            transforms.LoadImaged(
                keys=["image", "label", "sdf", "boundary"], channel_dim=0),
            transforms.EnsureChannelFirstd(keys=["image", "label", "sdf", "boundary"]),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=a_min, a_max=a_max,
                b_min=0.0, b_max=1.0, clip=True),
            transforms.RandRotated(
                keys=["image", "label", "sdf", "boundary"], 
                range_x=0.4, range_y=0.4,
                prob=0.5, keep_size=True, 
                mode=['bilinear', 'nearest', 'bilinear', 'nearest'],  align_corners=True),
            transforms.RandFlipd(
                keys=["image", "label", "sdf", "boundary"], 
                prob=0.1, spatial_axis=[0,1])
        ])

    val_transforms = transforms.Compose(
        [
            transforms.LoadImaged(
                keys=["image", "label", "sdf", "boundary"], channel_dim=0),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=a_min, a_max=a_max,
                b_min=0.0, b_max=1.0, clip=True),
        ])

    
    # define cachedataset and dataloader for training and validation
    train_ds = CacheDataset(
        data=train_files, transform=train_transforms,
        cache_rate=1.0, num_workers=num_workers)
    train_loader = DataLoader(
        train_ds, batch_size=train_batchsize, 
        shuffle=True, num_workers=num_workers)
    
    val_ds = CacheDataset(
        data=val_files, transform=val_transforms, 
        cache_rate=1.0, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=num_workers)

    return train_loader, val_loader

def post_transform(k):
    post_pred = transforms.Compose([
        transforms.AsDiscrete(argmax=True, to_onehot=k)])
    post_label = transforms.Compose([
        transforms.AsDiscrete(to_onehot=k)])
    
    post_trans = {
        'pred' : post_pred,
        'label' : post_label}

    return post_trans


def get_pinns2d_loader(
    traindata_dir, 
    train_batchsize=1, 
    num_workers=4,
    sample_idx=50):

    train_boundaries = sorted(
        glob.glob(os.path.join(traindata_dir, "boundariesTr", "*.npz")))
    train_sdfs = sorted(
        glob.glob(os.path.join(traindata_dir, "sdfsTr", "*.npz")))
    train_labels = sorted(
        glob.glob(os.path.join(traindata_dir, "labelsTr", "*.npz")))
    data_dicts = [
        {"boundary": boundary_name, "sdf":sdf_name, "label": label_name}
        for boundary_name, sdf_name, label_name in zip(train_boundaries, train_sdfs, train_labels)]
    
    train_files, val_files = data_dicts[sample_idx:sample_idx+1], data_dicts[sample_idx:sample_idx+1]
    train_transforms = transforms.Compose([
        transforms.LoadImaged(keys=["boundary", 'sdf', 'label'], channel_dim=0),])
    val_transforms = transforms.Compose([
        transforms.LoadImaged(keys=["boundary", 'sdf', 'label'], channel_dim=0),])

    train_ds = CacheDataset(
        data=train_files, transform=train_transforms,
        cache_rate=1.0, num_workers=num_workers)
    train_loader = DataLoader(
            train_ds, batch_size=train_batchsize, shuffle=True, num_workers=num_workers)
        
    val_ds = CacheDataset(
            data=val_files, transform=val_transforms, 
            cache_rate=1.0, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=num_workers)

    return train_loader, val_loader

def get_deeponet2d_loader(
    traindata_dir, 
    train_batchsize=16, 
    num_train = 500,
    num_workers=8):

    train_boundaries = sorted(
        glob.glob(os.path.join(traindata_dir, "boundariesTr", "*.npz")))
    train_labels = sorted(
        glob.glob(os.path.join(traindata_dir, "labelsTr", "*.npz")))
    train_sdfs = sorted(
        glob.glob(os.path.join(traindata_dir, "sdfsTr", "*.npz")))
    train_cnts = sorted(
        glob.glob(os.path.join(traindata_dir, "cntsTr", "*.npz")))
    data_dicts = [
        {"boundary": boundary_name, "cnt": cnt_name, 'sdf': sdf_name, 'label': label_name}
        for boundary_name, label_name, sdf_name, cnt_name in zip(train_boundaries, train_labels, train_sdfs, train_cnts)]
    
    train_files, val_files = data_dicts[:num_train], data_dicts[num_train:]
    train_transforms = transforms.Compose([transforms.LoadImaged(keys=["boundary", 'label', 'sdf', 'cnt'], channel_dim=0),])
    val_transforms = transforms.Compose([transforms.LoadImaged(keys=["boundary", 'label', 'sdf', 'cnt'], channel_dim=0),])

    train_ds = CacheDataset(
        data=train_files, transform=train_transforms,
        cache_rate=1.0, num_workers=num_workers)
    train_loader = DataLoader(
            train_ds, batch_size=train_batchsize, shuffle=True, num_workers=num_workers)
        
    val_ds = CacheDataset(
            data=val_files, transform=val_transforms, 
            cache_rate=1.0, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=num_workers)

    return train_loader, val_loader

def get_pinns3d_loader(
    traindata_dir, 
    train_batchsize=1, 
    num_workers=4,
    sample_idx=40):

    train_boundaries = sorted(
        glob.glob(os.path.join(traindata_dir, "boundariesTr", "*.npz")))
    train_sdfs = sorted(
        glob.glob(os.path.join(traindata_dir, "sdfsTr", "*.npz")))

    data_dicts = [
        {"boundary": boundary_name, "sdf": sdf_name}
        for boundary_name, sdf_name in zip(train_boundaries, train_sdfs)]

    train_files, val_files = data_dicts[sample_idx:sample_idx+1], data_dicts[sample_idx:sample_idx+1]

        # setup transforms for training and validation 
            # setup transforms for training and validation 
    train_transforms = transforms.Compose(
        [transforms.LoadImaged(
                keys=["boundary", 'sdf'], channel_dim=0),])

    val_transforms = transforms.Compose(
        [transforms.LoadImaged(
                keys=["boundary", 'sdf'], channel_dim=0),])
    
    # define cachedataset and dataloader for training and validation
    train_ds = CacheDataset(
        data=train_files, transform=train_transforms,
        cache_rate=1.0, num_workers=num_workers)
    train_loader = DataLoader(
        train_ds, batch_size=train_batchsize, 
        shuffle=True, num_workers=num_workers)
    
    val_ds = CacheDataset(
        data=val_files, transform=val_transforms, 
        cache_rate=1.0, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=num_workers)

    return train_loader, val_loader

if __name__ == '__main__':
    
    train_loader, val_loader = get_unet2d_loader(
        traindata_dir='/dataset/CXR/leftlungSZPreprocess_64x64/',
        valdata_dir='/dataset/CXR/leftlungSZPreprocess_128x128/',
        sample=True)

    sample_train_batch = next(iter(train_loader))
    sample_val_batch = next(iter(val_loader))

    print('tra - image : ', sample_train_batch['image'].shape)
    print('tra - label : ', sample_train_batch['label'].shape)
    print('tra - sdf : ', sample_train_batch['sdf'].shape)
    print('tra - boundary : ', sample_train_batch['boundary'].shape)

    print('val - image : ', sample_val_batch['image'].shape)
    print('val - label : ', sample_val_batch['label'].shape)
    print('val - sdf : ', sample_val_batch['sdf'].shape)
    print('val - boundary : ', sample_val_batch['boundary'].shape)
