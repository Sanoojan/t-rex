import os, sys
import torch
import torchvision
from torch import nn
import timm

sys.path.append(os.getcwd())
import utilities.runUtils as rutl
import utilities.logUtils as lutl


## Neural Net
class BarlowWrapnet(nn.Module):
    def __init__(self, args, basemodel):
        super().__init__()
        rutl.START_SEED()

        self.args = args
        self.basemodel = basemodel

        # Projector
        sizes = [self.basemodel.feat_outsize] + list(args.barlow_projector)
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)
        self.lastbn = nn.BatchNorm1d(sizes[-1], affine=False)

    def forward(self, x):
        feat = self.basemodel.backbone(x)
        res = self.basemodel.feat_dropout(feat)
        out = self.basemodel.classifier(res)
        return out, feat

    def forward_barlow(self, y1, y2):

        out1, feat1 = self.forward(y1)
        out2, feat2 = self.forward(y2)

        z1 = self.projector(feat1)
        z2 = self.projector(feat2)

        # empirical cross-correlation matrix
        ccorr = self.lastbn(z1).T @ self.lastbn(z2)

        # sum the cross-correlation matrix between all gpus
        ccorr.div_(self.args.batch_size)
        # torch.distributed.all_reduce(c)

        #rest of logic in Loss func

        return out1, out2, ccorr



## Loss func

CE_loss = nn.CrossEntropyLoss() 
def lossCEwithBT(pred1, pred2, ccorr, tgt):

    lambd = 0.0051

    on_diag = torch.diagonal(ccorr).add_(-1).pow_(2).sum()
    off_diag = off_diagonal(ccorr).pow_(2).sum()
    btloss = on_diag + lambd * off_diag

    ce1 = CE_loss(pred1, tgt)
    ce2 = CE_loss(pred2, tgt)

    return ce1 + ce2 + btloss

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


if __name__ == "__main__":

    from torchsummary import summary

    model = torchvision.models.resnet50()
    model = ('inception_v4')
    summary(model, (3, 299, 299))