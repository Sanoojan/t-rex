""" Classifier Network trainig
"""

import argparse
import json
import math
import os
import random
import signal
import subprocess
import sys
import time

from tqdm.autonotebook import tqdm
import PIL.Image
from torch import nn, optim
import torch

sys.path.append(os.getcwd())
import utilities.runUtils as rutl
import utilities.logUtils as lutl
from utilities.metricUtils import MultiClassMetrics
from algorithms.resnet import ClassifierNet
import datacode.classifier_data as ClsData
from algorithms.visiontransformer import vit_small_patch16_224


print(f"Pytorch version: {torch.__version__}")
print(f"cuda version: {torch.version.cuda}")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device Used:", device)

###============================= Configure and Setup ===========================

# #for ResNet
# cfg = rutl.ObjDict(
# dataset = "air", # "air+car", "food", "car"
# balance_class = False, #to be implemented

# epochs= 1000,
# batch_size= 64,
# workers= 4,
# learning_rate= 1e-4,
# weight_decay= 1e-6,

# feature_extract = "resnet50", # "resnet34/50/101"
# featx_pretrain = "DEFAULT",  # path-to-weights or None or DEFAULT-->imagenet
# featx_dropout = 0.0,
# classifier = [1024,], #First & Last MLPs will be set in code based on class out of dataset and FeatureExtractor
# clsfy_dropout = 0.5,

# checkpoint_dir= "hypotheses/res50-air/",
# restart_training=False
# )

#for ViT
cfg = rutl.ObjDict(
dataset = "food", # "air+car", "food", ## "car"
balance_class = False, #to be implemented

epochs= 1000,
batch_size= 100,
workers= 4,
learning_rate= 5e-5,
weight_decay= 0.001,

feature_extract = "resnet50", # "resnet34/50/101"
featx_pretrain = "DEFAULT",  # path-to-weights or None or DEFAULT-->imagenet
featx_dropout = 0.0,
classifier = [1024,], #First & Last MLPs will be set in code based on class out of dataset and FeatureExtractor
clsfy_dropout = 0.5,

checkpoint_dir= "hypotheses/vit_small-food/",
restart_training=True
)

### -----
parser = argparse.ArgumentParser(description='Classification task')
parser.add_argument('--load_json', type=str, metavar='JSON',
    help='Load settings from file in json format which override values hard codes in py file.')

args = parser.parse_args()

if args.load_json:
    with open(args.load_json, 'rt') as f:
        cfg.__dict__.update(json.load(f))

### ----------------------------------------------------------------------------
cfg.gLogPath = cfg.checkpoint_dir
cfg.gWeightPath = cfg.checkpoint_dir + '/weights/'

### ============================================================================


def getDatasetSelection():
    aircraftsdata_path = "/nfs/users/ext_sanoojan.baliah/Sanoojan/data/fgvc-aircraft-2013b/"
    foodxdata_path =  "/nfs/users/ext_sanoojan.baliah/Sanoojan/data/FoodX/food_dataset/"
    carsdata_path =  "/nfs/users/ext_sanoojan.baliah/Sanoojan/data/stanford_cars/"

    if cfg.dataset == "air":
        trainloader, train_info = ClsData.getAircraftsLoader( aircraftsdata_path,
                            batch_size=cfg.batch_size, workers =cfg.workers,
                            type_="train" )
        validloader, valid_info = ClsData.getAircraftsLoader( aircraftsdata_path,
                            batch_size=cfg.batch_size, workers =cfg.workers,
                            type_="valid" )
    elif cfg.dataset == "car":
        trainloader, train_info = ClsData.getCarsLoader( carsdata_path,
                            batch_size=cfg.batch_size, workers =cfg.workers,
                            type_="train" )
        validloader, valid_info = ClsData.getCarsLoader(carsdata_path,
                            batch_size=cfg.batch_size, workers =cfg.workers,
                            type_="valid" )

    elif cfg.dataset == "food":
        trainloader, train_info = ClsData.getFoodxLoader( foodxdata_path,
                            batch_size=cfg.batch_size, workers =cfg.workers,
                            type_="train" )
        validloader, valid_info = ClsData.getFoodxLoader( foodxdata_path,
                            batch_size=cfg.batch_size, workers =cfg.workers,
                            type_="valid" )

    elif cfg.dataset == "air+car":
        trainloader, train_info = ClsData.getAircraftsAndCarsLoader(
                            [aircraftsdata_path, carsdata_path],
                            batch_size=cfg.batch_size, workers =cfg.workers,
                            type_="train" )
        validloader, valid_info = ClsData.getAircraftsAndCarsLoader(
                            [aircraftsdata_path, carsdata_path],
                            batch_size=cfg.batch_size, workers =cfg.workers,
                            type_="valid" )
    else:
        raise ValueError("Unknown Dataset indicator set")

    lutl.LOG2DICTXT(["Train-",train_info], cfg.gLogPath +'/misc.txt')
    lutl.LOG2DICTXT(["Valid-", valid_info], cfg.gLogPath +'/misc.txt')

    return trainloader, validloader, len(train_info["Classes"])


def simple_main():

    ### SETUP
    rutl.START_SEED()
    gpu = 0
    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True

    if os.path.exists(cfg.checkpoint_dir) and (not cfg.restart_training):
        raise Exception("CheckPoint folder already exists and restart_training not enabled; Somethings Wrong!")
    if not os.path.exists(cfg.gWeightPath): os.makedirs(cfg.gWeightPath)

    with open(cfg.gLogPath+"/exp_cfg.json", 'a') as f:
        json.dump(vars(cfg), f, indent=4)


    ### DATA ACCESS
    trainloader, validloader, class_size  = getDatasetSelection()
    cfg.classifier.append(class_size) #Adding last layer of MLP

    ### MODEL, OPTIM

    #Resnet model
    # model = ClassifierNet(cfg).cuda(gpu)
    # device=

    model=vit_small_patch16_224(pretrained=True)
    model.head = nn.Linear(384, class_size)
    model=model.to(device)
    lossfn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                        weight_decay=cfg.weight_decay)
    lutl.LOG2TXT(f"Parameters:{rutl.count_train_param(model)}", cfg.gLogPath +'/model-info.txt')

    ## Automatically resume from checkpoint if it exists and enabled
    if os.path.exists(cfg.gWeightPath +'/checkpoint.pth'):
        ckpt = torch.load(cfg.gWeightPath+'/checkpoint.pth',
                          map_location='cpu')
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        lutl.LOG2TXT(f"Restarting Training from EPOCH:{start_epoch} of {cfg.checkpoint_dir}",  cfg.gLogPath +'/misc.txt')
    else:
        start_epoch = 0


    ### MODEL TRAINING
    
    start_time = time.time()
    best_acc = 0 ; best_loss = float('inf')
    trainMetric = MultiClassMetrics(cfg.gLogPath)
    validMetric = MultiClassMetrics(cfg.gLogPath)

    # scaler = torch.cuda.amp.GradScaler() # for mixed precision
    for epoch in range(start_epoch, cfg.epochs):

        ## ---- Training Routine ----
        model.train()
        
        for img, tgt in tqdm(trainloader):
            img = img.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            optimizer.zero_grad()
            ## with mixed precision
            # with torch.cuda.amp.autocast():
            pred = model.forward(img)
            loss = lossfn(pred, tgt)
            ## END with
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            loss.backward()
            optimizer.step()
            trainMetric.add_entry(torch.argmax(pred, dim=1), tgt, loss)

        ## save checkpoint states
        state = dict(epoch=epoch + 1, model=model.state_dict(),
                        optimizer=optimizer.state_dict())
        torch.save(state, cfg.gWeightPath +'/checkpoint.pth')


        ## ---- Validation Routine ----
        model.eval()
        with torch.no_grad():
            for img, tgt in tqdm(validloader):
                img = img.to(device, non_blocking=True)
                tgt = tgt.to(device, non_blocking=True)
                # with torch.cuda.amp.autocast():
                pred = model.forward(img)
                loss = lossfn(pred, tgt)
                ## END with
                validMetric.add_entry(torch.argmax(pred, dim=1), tgt, loss)

        ## Log Metrics
        stats = dict( epoch=epoch, time=int(time.time() - start_time),
                    trainloss = trainMetric.get_loss(),
                    trainacc = trainMetric.get_accuracy(),
                    validloss = validMetric.get_loss(),
                    validacc = validMetric.get_accuracy(), )
        lutl.LOG2DICTXT(stats, cfg.gLogPath+'/train-stats.txt')


        ## save best model #NOTE: Set model object based on architecture
        best_flag = False
        if stats['validacc'] > best_acc:
            torch.save(model.state_dict(), cfg.gWeightPath +'/bestmodel.pth')
            best_acc = stats['validacc']
            best_loss = stats['validloss']
            best_flag = True

        ## Log detailed validation
        detail_stat = dict( epoch=epoch, time=int(time.time() - start_time),
                            best = best_flag,
                            validreport =  validMetric.get_class_report() )
        lutl.LOG2DICTXT(detail_stat, cfg.gLogPath+'/validation-details.txt', console=False)

        trainMetric.reset()
        validMetric.reset(best_flag)

if __name__ == '__main__':
    simple_main()