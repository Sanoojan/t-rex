""" Classifier Network training with Barlow twin Correlation on features
"""

import os, sys, time
import argparse
import json
import random, math
import signal, subprocess

from tqdm.autonotebook import tqdm
import PIL.Image

import torch
from torch import nn, optim
import timm.loss

sys.path.append(os.getcwd())
import utilities.runUtils as rutl
import utilities.logUtils as lutl
from utilities.metricUtils import MultiClassMetrics

# from algorithms.resnet import ClassifierNet
from algorithms.convnext import ClassifierNet
# from algorithms.inception import ClassifierNet
from algorithms.barlowtwin import BarlowWrapnet, lossCEwithBT


from datacode.classifier_data import SimplifiedLoader


print(f"Pytorch version: {torch.__version__}")
print(f"cuda version: {torch.version.cuda}")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device Used:", device)

##============================= Configure and Setup ============================

cfg = rutl.ObjDict(
dataset = "air", # "air+car", "food", "car"
balance_class = False, #to be implemented

epochs= 1000,
batch_size= 64,
workers= 4,
learning_rate= 1e-4,
weight_decay= 1e-6,
stratergy = "BARLOW", #DEFAULT or AUGMIX or BARLOW
# augument= "DEFAULT", #

feature_extract = "convnext-base", #"convnext-tiny/small/base"
featx_pretrain = "IMGNET-22K",  # path-to-weights or None or DEFAULT-->imagenet
featx_dropout = 0.1,
classifier = [512,], #First & Last MLPs will be set in code based on class out of dataset and FeatureExtractor
clsfy_dropout = 0.5,

barlow_projector = [4096, 4096, 4096],

checkpoint_dir= "hypotheses/res50-air/",
restart_training=False,
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

## Checks and Balances
if cfg.stratergy != "BARLOW":
    raise ValueError("This train file only supports Barlow based training use differnt file other usage")
##------

def getDatasetSelection():

    loaderObj = SimplifiedLoader(cfg.dataset)
    trainloader, train_info = loaderObj.get_data_loader(type_= "train",
                    batch_size=cfg.batch_size, workers=cfg.workers,
                    augument= cfg.stratergy)

    validloader, valid_info = loaderObj.get_data_loader(type_= "valid",
                    batch_size=cfg.batch_size, workers=cfg.workers,
                    augument= "INFER")

    lutl.LOG2DICTXT({"Train-":train_info}, cfg.gLogPath +'/misc.txt')
    lutl.LOG2DICTXT({"Valid-": valid_info}, cfg.gLogPath +'/misc.txt')

    return trainloader, validloader, len(train_info["Classes"])


def getLossSelection():
    valid_loss = nn.CrossEntropyLoss()
    train_loss = lossCEwithBT

    return train_loss, valid_loss



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
    basemodel = ClassifierNet(cfg).cuda(gpu)
    model = BarlowWrapnet(cfg, basemodel).cuda(gpu)

    lossfn, v_lossfn = getLossSelection()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                        weight_decay=cfg.weight_decay)
    lutl.LOG2TXT(f"Parameters:{rutl.count_train_param(model)}", cfg.gLogPath +'/model-info.txt')

    ## Automatically resume from checkpoint if it exists and enabled
    if os.path.exists(cfg.gWeightPath +'/checkpoint.pth') and cfg.restart_training:
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
        for (img1, img2), tgt in tqdm(trainloader):
            img1 = img1.to(device, non_blocking=True)
            img2 = img2.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            ## with mixed precision
            # with torch.cuda.amp.autocast():
            p1, p2, cr = model.forward_barlow(img1, img2)
            loss = lossfn(p1, p2, cr, tgt)
            ## END with
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            loss.backward()
            optimizer.step()
            trainMetric.add_entry(torch.argmax((p1+p2)/2, dim=1), tgt, loss)

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
                pred = basemodel.forward(img)  ## or pred, _ = model.forward(img)
                loss = v_lossfn(pred, tgt)
                ## END with
                validMetric.add_entry(torch.argmax(pred, dim=1), tgt, loss)

        ## Log Metrics
        stats = dict( epoch=epoch, time=int(time.time() - start_time),
                    trainloss = trainMetric.get_loss(),
                    trainacc = trainMetric.get_accuracy(),
                    validloss = validMetric.get_loss(),
                    validacc = validMetric.get_accuracy(), )
        lutl.LOG2DICTXT(stats, cfg.gLogPath+'/train-stats.txt')


        ## save best model # TODO: add direct model saving
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