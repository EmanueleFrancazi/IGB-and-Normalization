import logging

import torch

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler

from torchvision.datasets import ImageFolder



logger = logging.getLogger(__name__)

def get_loader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    shift_const = 4  # Example constant value

    # Original mean and std values
    original_mean = [0.5, 0.5, 0.5]
    original_std = [0.5, 0.5, 0.5]

    # New mean values after adding the constant
    new_mean = [m + shift_const for m in original_mean]

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=new_mean, std=original_std),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=new_mean, std=original_std),
    ])
    
    
    # Data transforms (normalization & data augmentation)
    stats = ((0.4914 + shift_const, 0.4822 + shift_const, 0.4465 + shift_const), (0.2023, 0.1994, 0.2010))
    
    train_transform = []
    
    train_tfms = transforms.Compose([transforms.RandomCrop(32, padding=4, padding_mode='reflect'), 
                             transforms.RandomHorizontalFlip(), 
                             transforms.ToTensor(), 
                             transforms.Normalize(*stats,inplace=True)])
    
    valid_tfms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*stats)])

    

    if args.dataset == "cifar10":
        trainset = datasets.CIFAR10(root="./data",
                                    train=True,
                                    download=True,
                                    transform=transform_train)
        testset = datasets.CIFAR10(root="./data",
                                   train=False,
                                   download=True,
                                   transform=transform_test) if args.local_rank in [-1, 0] else None

    elif args.dataset == "CatsVsDogs":
        
        data_dir='/home/EAWAG/francaem/restored/data/Cifar10_Kaggle_link/Cats_Dogs'
        trainset = ImageFolder(data_dir+'/train', train_tfms)
        testset = ImageFolder(data_dir+'/test', valid_tfms)

    else:
        trainset = datasets.CIFAR100(root="./data",
                                     train=True,
                                     download=True,
                                     transform=transform_train)
        testset = datasets.CIFAR100(root="./data",
                                    train=False,
                                    download=True,
                                    transform=transform_test) if args.local_rank in [-1, 0] else None
    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=4,
                              pin_memory=True)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=4,
                             pin_memory=True) if testset is not None else None
    
    
    data = {}
    data['train_loader'] = train_loader
    data['test_loader'] = test_loader
    data['num_trdata_points'] =  len(trainset)
    data['num_valdata_points'] =  len(testset)

    return data
