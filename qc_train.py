import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import cv2
import glob
import os
import re
import shutil
import timm
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch import amp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import transformers
from sklearn.model_selection import StratifiedKFold, KFold, StratifiedGroupKFold, GroupKFold
import multiprocessing as mp
import segmentation_models_pytorch as smp
import copy
from collections import defaultdict
import gc
from tqdm import tqdm
import tifffile
from scipy import ndimage 
from colorama import Fore, Back, Style
from torch.autograd import Variable
c_  = Fore.GREEN
sr_ = Style.RESET_ALL

print("Importing DONE")

class CFG:
    seed = 0
    color_palette = [
        [255, 159,  19],
        [32, 177, 237],
        [41, 105, 255],
    ]
    batch_size = 4
    head = "DeepLabV3"
    backbone = 'resnet50'
    img_size = [512, 512]
    lr = 5e-4
    scheduler = 'CosineAnnealingLR'
    epochs = 40
    warmup_epochs = 2
    n_folds = 5
    folds_to_run = [4]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mode = 'binary'
    num_workers = mp.cpu_count()
    num_classes = 3
    n_accumulate = max(1, 4//batch_size)
    loss = 'BCE'
    optimizer = 'AdamW'
    weight_decay = 1e-6
    edge_smooth_value = 1
    edge_smooth_depth = 0
    output_dir = './deeplabv3_rn50__seed_0'
        
def prepare_loaders(df, fold):
    train_df = df.query("fold!=@fold").reset_index(drop=True)
    valid_df = df.query("fold==@fold").reset_index(drop=True)
    
    print(len(train_df))

    train_dataset = MyoVisionUS_Dataset(train_df, transforms=data_transforms['train'])
    valid_dataset = MyoVisionUS_Dataset(valid_df, transforms=data_transforms['valid'])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=CFG.batch_size,
                              num_workers=CFG.num_workers, shuffle=True, pin_memory=False)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=CFG.batch_size,
                              num_workers=CFG.num_workers, shuffle=False, pin_memory=False, drop_last=False)
    
    return train_loader, valid_loader

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def set_seed(seed = 42): #From https://www.kaggle.com/code/awsaf49/uwmgi-unet-train-pytorch/
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    print('> SEEDING DONE')
    
set_seed(CFG.seed)

class MyoVisionUS_Dataset(torch.utils.data.Dataset):
    def __init__(self, df, labeled=True, transforms=None, color_palette=CFG.color_palette):
        self.df = df
        self.labeled = labeled
        self.transforms = transforms
        self.color_palette = color_palette
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        
        row = self.df.iloc[index]
        img_path = row.img_path
        label_path = row.label_path

        img = np.array(cv2.imread(img_path))
        
        
        label = cv2.imread(label_path)
        temp = np.zeros((label.shape[0], label.shape[1], len(self.color_palette)))
            
        for i, color in enumerate(self.color_palette):
            temp[..., i][np.where((label==color).all(axis=2))] = CFG.edge_smooth_value
        
        label = temp
        del temp

        if self.transforms:
            data = self.transforms(image=img, mask=label)
            img  = np.transpose(data['image'], (2, 0, 1))
            label  = np.transpose(data['mask'], (2, 0, 1))
        
        if CFG.edge_smooth_depth != 0:
            mask =  ndimage.binary_erosion(label.tolist(), iterations=CFG.edge_smooth_depth)
            label[mask] = 1
            del mask
            
        return torch.tensor(img), torch.tensor(label)
    
data_transforms = {
    "train": A.Compose([
        A.Resize(*CFG.img_size, interpolation=cv2.INTER_NEAREST),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.5),
        A.RandomRotate90(p=0.5),
        A.OneOf([
            A.HueSaturationValue(10,15,10),
            A.CLAHE(clip_limit=2),
            A.RandomBrightnessContrast(),            
        ], p=0.4),
        A.Normalize(),
    ],),
    
    "valid": A.Compose([
        A.Resize(*CFG.img_size, interpolation=cv2.INTER_NEAREST),
        A.Normalize(),
        ], p=1.0),
}

JaccardLoss = smp.losses.JaccardLoss(mode=CFG.mode)
DiceLoss    = smp.losses.DiceLoss(mode=CFG.mode)
BCELoss     = smp.losses.SoftBCEWithLogitsLoss()
LovaszLoss  = smp.losses.LovaszLoss(mode=CFG.mode, per_image=False)
TverskyLoss = smp.losses.TverskyLoss(mode=CFG.mode, log_loss=False)
FocalLoss   = smp.losses.FocalLoss(mode=CFG.mode, alpha=0.25, gamma=2.0)

def dice_coef(y_true, y_pred, thr=0.5, dim=(2), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred>thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    dice = ((2*inter+epsilon)/(den+epsilon)).mean(dim=(1,0))
    return dice

def iou_coef(y_true, y_pred, thr=0.5, dim=(2), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred>thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    union = (y_true + y_pred - y_true*y_pred).sum(dim=dim)
    iou = ((inter+epsilon)/(union+epsilon)).mean(dim=(1,0))
    return iou

losses = {
    "Dice": DiceLoss,
    "Jaccard": JaccardLoss,
    "BCE": BCELoss,
    "Lovasz": LovaszLoss,
    "Tversky": TverskyLoss,
    "Focal": FocalLoss,
}

def get_scheduler(df, optimizer):
    
    if len(df[df['fold'] == CFG.folds_to_run[0]]) % CFG.batch_size != 0:
        num_steps = len(df[df['fold'] != CFG.folds_to_run[0]]) // CFG.batch_size + 1
    
    else:
        len(df[df['fold'] != CFG.folds_to_run[0]]) // CFG.batch_size
    
    if CFG.scheduler == 'CosineAnnealingLR':
        scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, CFG.warmup_epochs * num_steps, CFG.epochs * num_steps)
        
    elif CFG.scheduler == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=7, threshold=0.0001, min_lr=1e-6)
    elif CFG.scheduer == 'ExponentialLR':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.85)
        
    return scheduler

def get_optimizer(model, optimizer_name=CFG.optimizer):
    if CFG.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=CFG.lr)
    
    elif CFG.optimizer == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
        
    return optimizer

def build_model(pretrained="imagenet"):
    seg_models = {
        "Unet": smp.Unet,
        "Unet++": smp.UnetPlusPlus,
        "MAnet": smp.MAnet,
        "Linknet": smp.Linknet,
        "FPN": smp.FPN,
        "PSPNet": smp.PSPNet,
        "PAN": smp.PAN,
        "DeepLabV3": smp.DeepLabV3,
        "DeepLabV3+": smp.DeepLabV3Plus,
        "Segformer": smp.Segformer,
    }
    model = seg_models[CFG.head](
        encoder_name=CFG.backbone,      
        encoder_weights=pretrained,     
        in_channels=3,                  
        classes=CFG.num_classes,
        activation=None,
    )
    model.to(CFG.device)
    return model

def load_model(path):
    model = build_model()
    model.encoder.load_state_dict(torch.load(path)['state_dict'])
    model.eval()
    return model

def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    model.train()
    scaler = amp.GradScaler('cuda')
    
    dataset_size = 0
    running_loss = 0.0
    criterion = losses[CFG.loss]
    
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Train ')
    for step, (images, masks) in pbar:         
        images = images.to(device, dtype=torch.float)
        masks  = masks.to(device, dtype=torch.float)
        
        batch_size = images.size(0)
        
        with amp.autocast('cuda', enabled=True):
            y_pred = model(images)
            loss   = criterion(y_pred, masks)
            
        scaler.scale(loss).backward()
    
        if (step + 1) % CFG.n_accumulate == 0:
            scaler.step(optimizer)
            scaler.update()

            # zero the parameter gradients
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()
                
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(train_loss=f'{epoch_loss:0.4f}',
                        lr=f'{current_lr:0.5f}',
                        gpu_mem=f'{mem:0.2f} GB')
    torch.cuda.empty_cache()
    gc.collect()
    
    return epoch_loss


def valid_one_epoch(model, dataloader, device, epoch):
    with torch.no_grad():
        model.eval()
        
        dataset_size = 0
        running_loss = 0.0
        criterion = losses[CFG.loss]
        
        val_scores = []
        
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Valid ')
        for step, (images, masks) in pbar:        
            images  = images.to(device, dtype=torch.float)
            masks   = masks.to(device, dtype=torch.float)
            
            batch_size = images.size(0)
            
            y_pred  = model(images)
            loss    = criterion(y_pred, masks)
            
            running_loss += (loss.item() * batch_size)
            dataset_size += batch_size
            
            epoch_loss = running_loss / dataset_size
            
            y_pred = nn.Sigmoid()(y_pred)
            val_dice = []
            val_jaccard = []
            for i in range(CFG.num_classes):
                val_dice.append(dice_coef(masks[:, i], y_pred[:, i]).cpu().detach().numpy()*100.0)
                val_jaccard.append(iou_coef(masks[:, i], y_pred[:, i]).cpu().detach().numpy()*100.0)
            
            val_scores.append([val_dice, val_jaccard])
            
            mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
            pbar.set_postfix(valid_loss=f'{epoch_loss:0.4f}',
                            gpu_memory=f'{mem:0.2f} GB')
        val_scores  = np.mean(val_scores, axis=0)
        torch.cuda.empty_cache()
        gc.collect()
        
        return epoch_loss, val_scores
    
def run_training(model, optimizer, scheduler, train_loader, valid_loader, device, num_epochs, fold):
    if torch.cuda.is_available():
        print("cuda: {}\n".format(torch.cuda.get_device_name()))
    print("Model Parameters: {}\n".format(count_parameters(model)))
    
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_scores = -np.inf
    best_dice = -np.inf
    best_jaccard = -np.inf
    best_epoch = -1
    history = defaultdict(list)
    
    for epoch in range(1, num_epochs + 1): 
        gc.collect()
        print(f'Epoch {epoch}/{num_epochs}', end='')
        train_loss = train_one_epoch(model, optimizer, scheduler, 
                                           dataloader=train_loader, 
                                           device=CFG.device, epoch=epoch)
        
        val_loss, val_scores = valid_one_epoch(model, valid_loader, 
                                                 device=CFG.device, 
                                                 epoch=epoch)
        val_dice, val_jaccard = val_scores
        val_mean_dice = np.mean(val_dice)
        val_mean_jaccard = np.mean(val_jaccard)
    
        history['Train Loss'].append(train_loss)
        history['Valid Loss'].append(val_loss)
        history['Valid Dice'].append(val_dice)
        history['Valid Mean Dice'].append(val_mean_dice)
        history['Valid Jaccard'].append(val_jaccard)
        history['Valid Mean Jaccard'].append(val_mean_jaccard)
        
        print(f'Valid Dice: {val_dice} | Valid Jaccard: {val_jaccard}')
        
        # deep copy the model
        if (val_mean_dice+val_mean_jaccard)/2 >= best_scores:
            print(f"{c_}Valid Score Improved ({best_scores:0.4f} ---> {(val_mean_dice+val_mean_jaccard)/2:0.4f})")
            best_scores = (val_mean_dice+val_mean_jaccard)/2
            best_dice = val_dice
            best_jaccard = val_jaccard
            best_epoch   = epoch
            best_model_wts = copy.deepcopy(model.state_dict())
            PATH = f"{CFG.output_dir}/best_epoch-{fold:02d}.bin"
            torch.save(model.state_dict(), PATH)
            # Save a model file from the current directory
            print(f"Model Saved{sr_}")
            
        last_model_wts = copy.deepcopy(model.state_dict())
        PATH = f"{CFG.output_dir}/last_epoch-{fold:02d}.bin"
        torch.save(model.state_dict(), PATH)
            
        print(); print()
    
    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best Scores: {}, {} in Epoch {}".format(best_dice, best_jaccard, best_epoch))
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, history

def main():
    df = pd.DataFrame(glob.glob("dataset_path/*.png"), columns=["label_path"])
    df['base_path'] = df['label_path'].apply(lambda x: x.split('/')[-1])
    df['img_path'] = df['label_path'].apply(lambda x:  x.replace('labels', ''))
    df['group'] = df['base_path'].apply(lambda x: x[:-5])

    kf = KFold(n_splits=CFG.n_folds)
    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        df.loc[val_idx, 'fold'] = fold
    
    os.system(f'mkdir {CFG.output_dir}')
    for fold in CFG.folds_to_run:
        print(f'#'*15)
        print(f'### Fold: {fold}')
        print(f'#'*15)
        train_loader, valid_loader = prepare_loaders(df, fold=fold)
        model = build_model()
        optimizer = get_optimizer(model)
        scheduler = get_scheduler(df, optimizer)
        model, history = run_training(model, optimizer, scheduler, train_loader, valid_loader, 
                                      CFG.device, CFG.epochs, fold)

        log_df = pd.DataFrame.from_dict(history)
        log_df.to_csv(f'{CFG.output_dir}/fold_{fold}_log.csv', index=False)

        trace = torch.jit.trace(model, torch.randn(1, 3, 512, 512).to(CFG.device))
        torch.jit.save(trace, f'{CFG.output_dir}/fold_{fold}_best_model.pt')
    
if __name__ == "__main__":
    main()