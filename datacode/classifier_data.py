import os, sys, time
import random
import numpy as np
import pandas as pd
import scipy.io
from PIL import Image, ImageOps, ImageFilter

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets.folder import default_loader as tv_image_loader
from torch.utils.data import ConcatDataset

import timm.data

## ===================== Dataset Classes =======================================

class FGVCAircraft(torchvision.datasets.VisionDataset):
    """
    FGVC-Aircraft <http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/>`_ Dataset.

    Args:
        data_dir: "datasets/fgvc-aircraft-2013b/"
        class_type (string, optional): choose from ('variant', 'family', 'manufacturer').
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    class_types = ('variant', 'family', 'manufacturer')
    splits = ('train', 'val', 'trainval', 'test')

    def __init__(self, data_dir, class_type='variant', mode = "train", offset_class = 0,
                    transform=None, target_transform=None):
        super(FGVCAircraft, self).__init__(data_dir, transform=transform, target_transform=target_transform)

        if mode == "train": split = "trainval"
        elif mode == "valid": split = "test"
        if split not in self.splits:
            raise ValueError(f'Split "{split}" not found. Valid splits are: {self.splits}')
        if class_type not in self.class_types:
            raise ValueError(f'Class type "{class_type}" not found. Valid class types are: {self.class_types}')

        self.data_dir =  data_dir
        self.class_type = class_type
        self.split = split
        self.offset_class = offset_class

        (image_ids, targets, classes, class_to_idx) = self.find_classes()
        samples = self.make_dataset(image_ids, targets)

        self.samples = samples
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = tv_image_loader(path)

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target

    def __len__(self):
        return len(self.samples)

    def find_classes(self):
        # read classes file, separating out image IDs and class names
        classes_file = os.path.join(self.data_dir, 'data',
                            f'images_{self.class_type}_{self.split}.txt')

        image_ids = []
        targets = []
        with open(classes_file, 'r') as f:
            for line in f:
                split_line = line.strip().split(' ')
                image_ids.append(split_line[0])
                targets.append(' '.join(split_line[1:]))

        # index class names
        classes = np.unique(targets)
        class_to_idx = {classes[i]: i+self.offset_class for i in range(len(classes))}
        targets = [class_to_idx[c] for c in targets]

        return image_ids, targets, classes, class_to_idx

    def make_dataset(self, image_ids, targets):
        assert (len(image_ids) == len(targets))
        samples = []
        for i in range(len(image_ids)):
            item = (os.path.join(self.data_dir, 'data', 'images',
                                 f'{image_ids[i]}.jpg' ), targets[i])
            samples.append(item)
        return samples



class StanfordCars(torchvision.datasets.VisionDataset):
    """
    Stanford Cars <https://ai.stanford.edu/~jkrause/cars/car_dataset.html>_ Dataset.

    Args:
        data_dir (string): Root directory of the dataset.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, data_dir, mode="train", offset_class=0,
                    transform=None, target_transform=None):
        super(StanfordCars, self).__init__(data_dir, transform=transform, target_transform=target_transform)

        if mode == "train": data_flag = 0 # in precompiled matrix
        elif mode == "valid": data_flag = 1
        else: raise ValueError("Invalid Mode set use either Train or Valid")

        self.data_dir =  data_dir
        self.offset_class = offset_class

        loaded_mat = scipy.io.loadmat(os.path.join(data_dir, 'cars_annos.mat'))
        class_names = loaded_mat['class_names'][0]
        loaded_mat = loaded_mat['annotations'][0]
        self.samples = []
        for item in loaded_mat:
            if data_flag == int(item[-1][0]):
                path = str(item[0][0])
                label = int(item[-2][0]) - 1
                # Modify class index as we are going to concat
                label = label + self.offset_class
                self.samples.append((path, label))

        self.class_to_idx = {class_names[i][0]: i+self.offset_class for i in range(len(class_names))}
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = tv_image_loader(os.path.join(self.data_dir, path))

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target

    def __len__(self):
        return len(self.samples)




class FoodXDataset(torch.utils.data.Dataset):
    """ data_dir: datasets/FoodX/food_dataset/
        mode: train or val
    """

    splits = ('train', 'val')
    def __init__(self, data_dir, mode = "train", offset_class = 0,
                    transform = None, target_transform = None,
                    ):

        if mode == "train": split = "train"
        elif mode =="valid": split = "val"
        if split not in self.splits:
            raise ValueError(f'Split "{split}" not found. Valid splits are: {self.splits}')

        image_folder = f'{data_dir}/{split}_set/'
        csv_path = f'{data_dir}/annot/{split}_info.csv'
        dataframe = pd.read_csv(csv_path, names= ['image_name','label'])
        dataframe['path'] = dataframe['image_name'].map(lambda x: os.path.join(image_folder, x))

        self.offset_class = offset_class
        class_list =[ ls.strip().split(" ") for ls in open(f"{data_dir}/annot/class_list.txt").readlines()]
        self.class_to_idx = { l[1]: int(l[0])+self.offset_class for l in class_list}

        self.dataframe = dataframe
        self.transform = transform
        self.target_transform = target_transform
        #self.samples =  ##TODO for future

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        image = tv_image_loader(row["path"])
        target = row['label']

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target


## ========================= Transforms ========================================

IMG_SIZE = 299


class InferTransforms:
    def __init__(self, infer=False):
        self.img_size =  IMG_SIZE
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize( mean=(0.485, 0.456, 0.406),
                                  std=(0.229, 0.224, 0.225)),
        ])
    def __call__(self, x):
        y = self.transform(x)
        return y
    def get_composition(self):
        return str(self.transform)


class ClassifyTransforms:
    def __init__(self, infer=False):
        self.img_size =  IMG_SIZE
        train_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            # transforms.RandomResizedCrop(self.img_size, scale=(0.6, 1.0),
            #             interpolation=transforms.InterpolationMode.BICUBIC),
            # transforms.RandomAffine(degrees=(-180, 180), translate=(0.2, 0.2),
            #             interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandAugment(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            # transforms.AugMix(),
            transforms.ToTensor(),
            transforms.Normalize( mean=(0.485, 0.456, 0.406),
                                  std=(0.229, 0.224, 0.225)),

        ])
        infer_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize( mean=(0.485, 0.456, 0.406),
                                  std=(0.229, 0.224, 0.225)),
        ])

        self.transform = infer_transform if infer else train_transform

    def __call__(self, x):
        y = self.transform(x)
        return y

    def get_composition(self):
        return str(self.transform)


class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class BarlowTransforms:
    def __init__(self):
        self.img_size =  IMG_SIZE
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size), 
                interpolation=transforms.InterpolationMode.BICUBIC),
            # transforms.RandomResizedCrop(self.img_size, 
            #   interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],  p=0.8 ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=1.0),
            Solarization(p=0.0),
            transforms.RandAugment(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.transform_prime = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size), 
                interpolation=transforms.InterpolationMode.BICUBIC),
            # transforms.RandomResizedCrop(self.img_size, 
            #   interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)], p=0.8 ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.1),
            Solarization(p=0.2),
            transforms.RandAugment(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2

    def get_composition(self):
        return str(self.transform) + str(self.transform_prime)

## ===================== Dataloaders Func ======================================


class SimplifiedLoader():
    def __init__(self, set_name) -> None:
        self.aircraftsdata_path = "/apps/local/shared/CV703/datasets/fgvc-aircraft-2013b/"
        self.foodxdata_path =  "/apps/local/shared/CV703/datasets/FoodX/food_dataset/"
        self.carsdata_path =  "/apps/local/shared/CV703/datasets/stanford_cars/"
        self.set_name = set_name
        self.data_info = {}


    def get_data_loader(self, type_, batch_size=64, workers=2, augument= "DEFAULT"):

        self.data_info["type"] = type_
        transform = self._fetch_transforms(type_, augument)
        dataset = self._fetch_dataset(self.set_name, type_, transform)


        loader = self._default_loader_impl(dataset,
                            batch_size=batch_size, workers=workers,
                            type_= type_)

        if augument == "AUGMIX": # override loader
            print("Loading Augmix based loader from timm !.!.!")
            loader = self._augmix_loader_impl(dataset,
                                batch_size=batch_size, workers=workers,
                                splits=3, type_=type_)

        return loader, self.data_info.copy()

    def _fetch_transforms(self, type_, augument):

        if (type_ in ["valid", "test", "infer"]) or (augument in ["INFER", None]):
            data_transform = InferTransforms()
        elif augument == "DEFAULT":
            data_transform = ClassifyTransforms()
        elif augument == "AUGMIX":
            data_transform = None # will be set by _augmix_loader_impl
        elif augument == "BARLOW":
            data_transform = BarlowTransforms()
        else:
            raise ValueError("Unknown augument specified")


        self.data_info["transform"] = data_transform.get_composition() if data_transform else None

        return data_transform


    def _fetch_dataset(self, set_name, type_, transform):
        infer_flag = False
        if type_ in ['valid', 'test', 'infer']:
            infer_flag = True

        if set_name == "air":
            dataset = FGVCAircraft(data_dir=self.aircraftsdata_path,
                                    mode=type_, transform=transform)
        elif set_name == "car":
           dataset = StanfordCars(data_dir=self.carsdata_path,
                                    mode=type_, transform=transform)
        elif set_name == "food":
           dataset = FoodXDataset(data_dir=self.foodxdata_path,
                                    mode=type_, transform=transform)
        elif set_name == "air+car":
            air_dataset = FGVCAircraft(data_dir=self.aircraftsdata_path,
                            mode=type_, transform=transform)
            car_dataset = StanfordCars(data_dir=self.carsdata_path,
                            mode=type_,
                            offset_class=len(air_dataset.class_to_idx),
                            transform=transform)
            dataset = ConcatDataset([air_dataset, car_dataset])
            dataset.class_to_idx = dict(list(air_dataset.class_to_idx.items()) +  \
                    list(car_dataset.class_to_idx.items())) #union operation
            dataset.transform = transform
        else:
            raise ValueError("Unknown Dataset indicator set")

        self.data_info.update({ "Classes": dataset.class_to_idx ,
                    "ClassesSize": len(dataset.class_to_idx) ,
                    "DatasetSize": dataset.__len__() })
        return dataset


    def _default_loader_impl(self, dataset, batch_size=64, workers=2, type_ = 'train',):
        shuffle_flag = True #important
        if type_ in ['valid', 'test', 'infer']:
            shuffle_flag = False

        loader = torch.utils.data.DataLoader( dataset,
            batch_size=batch_size, num_workers=workers, shuffle=shuffle_flag,
            pin_memory=True)

        return loader


    def _augmix_loader_impl(self, dataset, batch_size=64, workers=2,
                            splits = 3, type_ = "train"):
        ## TODO: figure out and Fix issue with concat dataset
        """ timm library based usage
        Reference:[1]https://github.com/rwightman/pytorch-image-models/blob/main/timm/data/loader.py#L189
                [*2] https://github.com/rwightman/pytorch-image-models/blob/d5aa17e41572ececee0f7829ec1640384532c5d2/timm/data/auto_augment.py#L951
        """
        assert type_ == "train",  "AUGMIX can be invoked only for training"

        dataset = timm.data.AugMixDataset(dataset, num_splits=splits)
        loader = timm.data.create_loader(dataset,
                    input_size=(3, 224, 224),
                    batch_size=batch_size,
                    is_training=True,
                    use_prefetcher=True,
                    scale=[0.08, 1.0],
                    ratio=[3./4., 4./3.],
                    hflip=0.5,
                    vflip=0.5,
                    color_jitter=0.8,
                    auto_augment="rand",  # refer [*2]
                    num_aug_splits=splits,
                    interpolation="bilinear",
                    mean=(0.485, 0.456, 0.406), ##Imagenet
                    std=(0.229, 0.224, 0.225),  ## Imagenet
                    num_workers=workers,
                    collate_fn=None, #FastCollateMixup
                    pin_memory=True,           )


        return loader


##------------------------------------------------------------------------------
## Naive Individual Loader TODO: Remove when refactoring

def getCifar100Loader(folder, batch_size, workers=2, type_ = 'train'):

    if type_ == 'train': trainset = True
    else: trainset = False
    dataset = torchvision.datasets.CIFAR100(folder, train=trainset, download=True,
                transform= transforms.ToTensor(), )
    cls_idx = dataset.class_to_idx
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=workers, shuffle=True,
        pin_memory=True)

    return loader, cls_idx


def getAircraftsLoader(folder, batch_size, workers=2,
                    type_ = 'train', augmix = False):

    infer_flag = False; shuffle_flag = True
    if type_ in ['valid', 'test', 'infer']:
        infer_flag = True; shuffle_flag = False


    data_transform = ClassifyTransforms(infer_flag)
    dataset = FGVCAircraft(data_dir=folder, mode=type_,
                    transform=data_transform)

    loader = torch.utils.data.DataLoader( dataset,
        batch_size=batch_size, num_workers=workers, shuffle=shuffle_flag,
        pin_memory=True)

    data_info = {"type": type_,
                "Classes": dataset.class_to_idx ,
                "DatasetSize": dataset.__len__(),
                "Transforms": str(data_transform.get_composition()) }

    return loader, data_info


def getCarsLoader(folder, batch_size, workers=2, type_ = 'train'):
    infer_flag = False; shuffle_flag = True
    if type_ in ['valid', 'test', 'infer']:
        infer_flag = True; shuffle_flag = False

    data_transform = ClassifyTransforms(infer_flag)
    dataset = StanfordCars(data_dir=folder, mode=type_,
                    transform=data_transform)

    loader = torch.utils.data.DataLoader( dataset,
        batch_size=batch_size, num_workers=workers, shuffle=shuffle_flag,
        pin_memory=True)

    data_info = {"type": type_,
                "Classes": dataset.class_to_idx ,
                "DatasetSize": dataset.__len__(),
                "Transforms": str(data_transform.get_composition()) }

    return loader, data_info


def getAircraftsAndCarsLoader(folders, batch_size, workers=2, type_ = 'train'):
    """ folders: [0] Aircrafts data path [1] Stanford cars data path
    """
    if not isinstance(folders, list) or len(folders)!=2:
        raise ValueError("Requires Two folders path for concatenation")

    infer_flag = False; shuffle_flag = True
    if type_ in ['valid', 'test', 'infer']:
        infer_flag = True; shuffle_flag = False

    data_transform = ClassifyTransforms(infer_flag)

    air_dataset = FGVCAircraft(data_dir=folders[0], mode=type_,
                    transform=data_transform)
    car_dataset = StanfordCars(data_dir=folders[1], mode=type_,
                    offset_class=len(air_dataset.class_to_idx),
                    transform=data_transform )

    dataset = ConcatDataset([air_dataset, car_dataset])

    class_to_idx = dict(list(air_dataset.class_to_idx.items()) +  \
                    list(car_dataset.class_to_idx.items())) #union operation
    loader = torch.utils.data.DataLoader( dataset,
        batch_size=batch_size, num_workers=workers, shuffle=shuffle_flag,
        pin_memory=True)

    data_info = {"type": type_,
                "Classes": class_to_idx ,
                "DatasetSize": dataset.__len__(),
                "Transforms": str(data_transform.get_composition()) }

    return loader, data_info


def getFoodxLoader(folder, batch_size, workers=2, type_ = 'train'):
    infer_flag = False; shuffle_flag = True
    if type_ in ['valid', 'test', 'infer']:
        infer_flag = True; shuffle_flag = False

    data_transform = ClassifyTransforms(infer_flag)
    dataset = FoodXDataset(data_dir=folder, mode=type_,
                    transform=data_transform)

    loader = torch.utils.data.DataLoader( dataset,
        batch_size=batch_size, num_workers=workers, shuffle=shuffle_flag,
        pin_memory=True)

    data_info = {"type": type_,
                "Classes": dataset.class_to_idx ,
                "DatasetSize": dataset.__len__(),
                "Transforms": str(data_transform.get_composition()) }

    return loader, data_info





##========================= DatasetChecking ====================================

import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image as tv_to_pil_image

def show_loaded_imgs(imgs):
    if not isinstance(imgs, list): imgs = [imgs]
    cols = 5
    rows = np.ceil(len(imgs) / 5).astype(int)
    fig = plt.figure(figsize=(10, 7))
    for i, img in enumerate(imgs):
        img = img.permute(1,2,0).cpu().numpy()
        img = (img - img.min()) / (img.max()-img.min())
        fig.add_subplot(rows,cols,i+1)
        plt.imshow(img)
        plt.axis('off')
    plt.show()

if __name__ == "__main__":

    aircraftsdata_path = "/apps/local/shared/CV703/datasets/fgvc-aircraft-2013b/"
    foodxdata_path =  "/apps/local/shared/CV703/datasets/FoodX/food_dataset/"
    carsdata_path =  "/apps/local/shared/CV703/datasets/stanford_cars/"


    # dataloader,_ = getAircraftsAndCarsLoader([aircraftsdata_path, carsdata_path],
                    # batch_size=2, type_="train")

    # dataloader,_ = getCarsLoader( carsdata_path, batch_size=2, type_="valid")

    dataloader, _ = SimplifiedLoader("car").get_data_loader(type_= "train",
                    batch_size=2, workers=2, augument= "BARLOW")


    ## -------------------- Plotting Loop---------------------------------------
    count = 5  ## SET THIS
    imgs = []; tgts = []

    # ridx  = np.random.choice(len(dataset), count)
    # for i in ridx:
        # img, tgt =  dataset.__getitem__(i)
        # imgs.append(img)
        # tgts.append(tgt)

    iter_dataloader = iter(dataloader)
    for i in range(count):
        img_ret, tgt =  next(iter_dataloader) ## ridx will be ignored
        for img in img_ret: # for handling image augumentation
            print(img.min(), img.max())
            img_ = [ img[i].squeeze() for i in range(img.shape[0]) ]
            tgt_ = [ tgt[i].squeeze() for i in range(tgt.shape[0]) ]
            imgs.extend(img_)
            tgts.extend(tgt_)

    show_loaded_imgs(imgs)











