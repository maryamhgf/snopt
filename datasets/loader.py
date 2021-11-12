import numpy as np
import torch
import os
from torchvision import transforms as T


from torchvision import transforms
from prefetch_generator import BackgroundGenerator
import util

import torchvision.datasets as torch_data
from .time_series import uea as uea_data
from .tabular import maf as maf_data


def _gen_mini_dataset(dataset, dataset_ratio):
    n_dataset = dataset.shape[0]
    n_mini_dataset = int(dataset_ratio*n_dataset)
    s = torch.from_numpy(np.random.choice(
        np.arange(n_dataset, dtype=np.int64), n_mini_dataset, replace=False)
    )
    return dataset[s]


class DataLoaderX(torch.utils.data.DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class TabularLoader:
    def __init__(self,opt, data, batch_size=None, shuffle=True):

        self.data_size = data.shape[0]
        self.opt = opt
        self.device = opt.device

        self.data = data.to(opt.device)
        self.batch_size = opt.batch_size if batch_size is None else batch_size
        self.shuffle = shuffle

        self.input_dim = data.shape[-1]
        self.output_dim= [data.shape[-1]]

        loc = torch.zeros(data.shape[-1]).to(opt.device)
        covariance_matrix = torch.eye(data.shape[-1]).to(opt.device) # TODO(Guan) scale down the cov ?
        self.p_z0 = torch.distributions.MultivariateNormal(loc=loc, covariance_matrix=covariance_matrix)
        self._reset_idxs()
        self.data_size = len(self.idxs_by_batch_size)

    def _reset_idxs(self):
        idxs = torch.randperm(self.data.shape[0]) if self.shuffle else torch.arange(self.data.shape[0])
        self.idxs_by_batch_size = idxs.split(self.batch_size)
        self.batch_idx = 0

    def __len__(self):
        return self.data_size

    def __iter__(self):
        return self

    def __next__(self):
        if self.batch_idx >= len(self.idxs_by_batch_size):
            self._reset_idxs()
            raise StopIteration

        s = self.idxs_by_batch_size[self.batch_idx]
        self.batch_idx += 1
        x = self.data[s]
        logp_diff_t1 = torch.zeros(x.shape[0], 1, device=x.device)
        return (x, logp_diff_t1), self.p_z0


def get_uea_loader(opt):

    print(util.magenta("loading uea data..."))

    dataset_name = {
        'CharT' :'CharacterTrajectories',
        'ArtWR' : 'ArticularyWordRecognition',
        'SpoAD' :     'SpokenArabicDigits',
    }.get(opt.problem)

    missing_rate = 0.0
    device = opt.device
    intensity_data = True

    (times, train_dataloader, val_dataloader,
     test_dataloader, num_classes, input_channels) = uea_data.get_data(dataset_name, missing_rate, device,
                                                                           intensity=intensity_data,
                                                                           batch_size=opt.batch_size)

    # we'll return dataloader and store the rest in opt
    opt.times = times
    opt.output_dim = num_classes
    opt.input_dim = input_channels
    return train_dataloader, test_dataloader


def get_tabular_loader(opt, test_batch_size=1000):
    assert opt.problem in ['gas', 'miniboone']
    print(util.magenta("loading tabular data..."))

    data = maf_data.get_data(opt.problem)
    data.trn.x = torch.from_numpy(data.trn.x)
    data.val.x = torch.from_numpy(data.val.x)
    data.tst.x = torch.from_numpy(data.tst.x)

    if opt.dataset_ratio < 1.0:
        data.trn.x = _gen_mini_dataset(data.trn.x, opt.dataset_ratio)
        data.val.x = _gen_mini_dataset(data.val.x, opt.dataset_ratio)
        data.tst.x = _gen_mini_dataset(data.tst.x, opt.dataset_ratio)

    train_loader = TabularLoader(opt, data.trn.x, shuffle=True)
    val_loader   = TabularLoader(opt, data.val.x, batch_size=test_batch_size, shuffle=False)
    test_loader  = TabularLoader(opt, data.tst.x, batch_size=test_batch_size, shuffle=False)

    opt.input_dim = train_loader.input_dim
    opt.output_dim = train_loader.output_dim

    return train_loader, test_loader

def generate_dataloader_ImageNet(data, name, batch_size, transform):
    if data is None: 
        return None
    
    # Read image files to pytorch dataset using ImageFolder, a generic data 
    # loader where images are in format root/label/filename
    # See https://pytorch.org/vision/stable/datasets.html
    if transform is None:
        dataset = datasets.ImageFolder(data, transform=T.ToTensor())
    else:
        dataset = datasets.ImageFolder(data, transform=transform)

    # Set options for device
    if use_cuda:
        kwargs = {"pin_memory": True, "num_workers": 1}
    else:
        kwargs = {}
    
    # Wrap image dataset (defined above) in dataloader 
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                        shuffle=(name=="train"), 
                        **kwargs)
    
    return dataloader

def get_img_loader(opt, test_batch_size=1000):
    print(util.magenta("loading image data..."))
    if(opt.problem == "tinyImageNet"):
        #run !wget http://cs231n.stanford.edu/tiny-imagenet-200.zip first to download data to your local macine.
        #run !unzip -qq 'tiny-imagenet-200.zip' to unzip the data.
        # Define main data directory
        DATA_DIR = 'tiny-imagenet-200' # Original images come in shapes of [3,64,64]

        # Define training and validation data paths
        TRAIN_DIR = os.path.join(DATA_DIR, 'train') 
        VALID_DIR = os.path.join(DATA_DIR, 'val')
        val_img_dir = os.path.join(VALID_DIR, 'images')

        # Open and read val annotations text file
        fp = open(os.path.join(VALID_DIR, 'val_annotations.txt'), 'r')
        data = fp.readlines()

        # Create dictionary to store img filename (word 0) and corresponding
        # label (word 1) for every line in the txt file (as key value pair)
        val_img_dict = {}
        for line in data:
            words = line.split('\t')
            val_img_dict[words[0]] = words[1]
        fp.close()
        
        for img, folder in val_img_dict.items():
            newpath = (os.path.join(val_img_dir, folder))
            if not os.path.exists(newpath):
                os.makedirs(newpath)
            if os.path.exists(os.path.join(val_img_dir, img)):
                os.rename(os.path.join(val_img_dir, img), os.path.join(newpath, img))
        preprocess_transform = T.Compose([
                        T.Resize(256), # Resize images to 256 x 256
                        T.CenterCrop(224), # Center crop image
                        T.RandomHorizontalFlip(),
                        T.ToTensor(),  # Converting cropped images to tensors
                        T.Normalize(mean=[0.485, 0.456, 0.406], 
                                    std=[0.229, 0.224, 0.225])
        ])
        # Define batch size for DataLoaders

        # Create DataLoaders for pre-trained models (normalized based on specific requirements)
        train_loader = generate_dataloader_ImageNet(TRAIN_DIR, "train", opt.batch_size,
                                        transform=preprocess_transform)

        test_loader = generate_dataloader_ImageNet(val_img_dir, "val", test_batch_size,
                                        transform=preprocess_transform)
        opt.input_dim = [3, 64, 64]
        opt.output_dim = 200       

    else:
        dataset_builder, root, input_dim, output_dim = {
            'mnist':   [torch_data.MNIST,  'data/img/mnist',  [1,28,28], 10],
            'SVHN':    [torch_data.SVHN,   'data/img/svhn',   [3,32,32], 10],
            'cifar10': [torch_data.CIFAR10,'data/img/cifar10',[3,32,32], 10],
            'cifar100': [torch_data.CIFAR100,'data/img/cifar100',[3,32,32], 100],
        }.get(opt.problem)
        opt.input_dim = input_dim
        opt.output_dim = output_dim

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        feed_dict = dict(download=True, root=root, transform=transform)
        train_dataset = dataset_builder(**feed_dict) if opt.problem=='SVHN' else dataset_builder(train=True, **feed_dict)
        test_dataset  = dataset_builder(**feed_dict) if opt.problem=='SVHN' else dataset_builder(train=False, **feed_dict)

        feed_dict = dict(num_workers=2, drop_last=True)
        train_loader = DataLoaderX(train_dataset, batch_size=opt.batch_size, shuffle=True, **feed_dict)
        test_loader  = DataLoaderX(test_dataset, batch_size=test_batch_size, shuffle=False, **feed_dict)

    return train_loader, test_loader