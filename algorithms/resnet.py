import os
import torch
import torchvision
from torch import nn
import utilities.runUtils as rutl
import utilities.logUtils as lutl


## Neural Net
class ClassifierNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        rutl.START_SEED()

        self.args = args

        # Feature Extractor
        self.backbone, out_size = self._load_resnet_backbone()
        self.feat_dropout = nn.Dropout(p=self.args.featx_dropout)

        # Classifier
        sizes = [out_size] + list(args.classifier)
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=self.args.clsfy_dropout))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))

        self.classifier = nn.Sequential(*layers)


    def forward(self, x):
        x = self.backbone(x)
        x = self.feat_dropout(x)
        out = self.classifier(x)

        return out


    def _load_resnet_backbone(self):

        torch_pretrain = "DEFAULT" if self.args.featx_pretrain == "DEFAULT" else None

        if self.args.feature_extract == 'resnet18':
            backbone = torchvision.models.resnet18(zero_init_residual=True,
                                 weights=torch_pretrain)
            outfeat_size = 512
        elif self.args.feature_extract == 'resnet34':
            backbone = torchvision.models.resnet34(zero_init_residual=True,
                                weights=torch_pretrain)
            outfeat_size = 512
        elif self.args.feature_extract == 'resnet50':
            backbone = torchvision.models.resnet50(zero_init_residual=True,
                                weights=torch_pretrain)
            outfeat_size = 2048

        elif self.args.feature_extract == 'resnet101':
            backbone = torchvision.models.resnet101(zero_init_residual=True,
                                weights=torch_pretrain)
            outfeat_size = 2048

        elif self.args.feature_extract == 'resnet152':
            backbone = torchvision.models.resnet152(zero_init_residual=True,
                                weights=torch_pretrain)
            outfeat_size = 2048

        backbone.fc = nn.Identity() #remove fc of default arc

        # pretrain from external file
        if os.path.exists(self.args.featx_pretrain):
            backbone = self._load_weights_from_file(backbone,
                                self.args.featx_pretrain )
            print("Loaded:", self.args.featx_pretrain )

        return backbone, outfeat_size



    def _load_weights_from_file(self, model, weight_path, flexible = True):
        def _purge(key): # hardcoded logic
            return key.replace("backbone.", "")

        model_dict = model.state_dict()
        weight_dict = torch.load(weight_path)

        if 'model' in weight_dict.keys(): pretrain_dict = weight_dict['model']
        else:   pretrain_dict = weight_dict

        pretrain_dict = { _purge(k) : v for k, v in pretrain_dict.items()}

        if flexible:
            pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
        if not len(pretrain_dict):
            raise Exception(f"No weight names match to be loaded; though file exits ! {weight_path}, Dict: {weight_dict.keys()}")

        lutl.LOG2TXT(f"Pretrained layers:{pretrain_dict.keys()}" ,
                        self.args.gLogPath+"/misc.txt", console=False )

        model_dict.update(pretrain_dict)
        model.load_state_dict(model_dict)

        return model



