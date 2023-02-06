


""" Classifier Network trainig
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
import timm
import timm.loss

sys.path.append(os.getcwd())
import utilities.runUtils as rutl
import utilities.logUtils as lutl
from utilities.metricUtils import MultiClassMetrics
from algorithms.visiontransformer import vit_small_patch16_224, vit_base_patch16_224, vit_base_patch16_224_in21k
# from algorithms.resnet import ClassifierNet
from algorithms.convnext import ClassifierNet
import algorithms.clip as clip

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
batch_size= 80,
workers= 4,
learning_rate= 1e-4,
weight_decay= 1e-6,
augument= "DEFAULT", #DEFAULT or AUGMIX

feature_extract = "inceptionV4", # "resnet34/50/101"  convnext-small
featx_pretrain = "DEFAULT",  # path-to-weights or None or DEFAULT-->imagenet
featx_dropout = 0.0,
classifier = [1024,], #First & Last MLPs will be set in code based on class out of dataset and FeatureExtractor
clsfy_dropout = 0.5,

checkpoint_dir= "hypotheses/inceptionv4-air/",
restart_training=True
)

# dataset_to_train="air"  # "air+car", "food", ## "car"
# cfg = rutl.ObjDict(
# dataset = dataset_to_train,
# checkpoint_dir= "hypotheses/base_21k"+dataset_to_train+"/",
# balance_class = False, #to be implemented

# epochs= 200,
# batch_size= 80,
# workers= 4,
# learning_rate= 5e-5,
# weight_decay= 0.0001,
# augument= "DEFAULT", #DEFAULT or AUGMIX,

# feature_extract = "vitbase", # "resnet34/50/101"
# featx_pretrain = "DEFAULT",  # path-to-weights or None or DEFAULT-->imagenet
# featx_dropout = 0.0,
# classifier = [1024,], #First & Last MLPs will be set in code based on class out of dataset and FeatureExtractor
# clsfy_dropout = 0.5,


# restart_training=True
# )

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

    loaderObj = SimplifiedLoader("air")
    trainloader, train_info = loaderObj.getDataLoader(type_= "train", 
                    batch_size=cfg.batch_size, workers=cfg.workers, 
                    augument= cfg.augument)

    validloader, valid_info = loaderObj.getDataLoader(type_= "valid", 
                    batch_size=cfg.batch_size, workers=cfg.workers,
                    augument= "DEFAULT")

    lutl.LOG2DICTXT(["Train-",train_info], cfg.gLogPath +'/misc.txt')
    lutl.LOG2DICTXT(["Valid-", valid_info], cfg.gLogPath +'/misc.txt')

    return trainloader, validloader, len(train_info["Classes"])


def getLossSelection():
    train_loss = valid_loss = nn.CrossEntropyLoss()
    if cfg.augument == "AUGMIX":
        train_loss = timm.loss.JsdCrossEntropy(num_splits=3)

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
    # model = ClassifierNet(cfg).cuda(gpu)

    #load inceptionV4 from timm
    model = timm.create_model('inception_v4', pretrained=True)

  

    # model=vit_base_patch16_224_in21k(pretrained=True)
    # model.head = nn.Linear(768, class_size)


    # clipname = 'ViT-B/16'
    # mod, preprocess = clip.load(clipname)
    
    # mod = mod.visual.float()
    # lin=torch.nn.Linear(768, class_size)
    # model=nn.Sequential(mod, lin)


    model=model.to(device)


    lossfn, v_lossfn = getLossSelection()
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
            with torch.cuda.amp.autocast():
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