"""Data loading utilities for CIFAR datasets."""
import os
import torch
from torch.utils.data import DataLoader, random_split
import numpy as np


def get_data_loaders(dataset='cifar10', batch_size=128, num_workers=4, sample_size=None, val_split=0.1, random_seed=42):
    """Get data loaders for training, validation, and test sets.
    
    Args:
        dataset: 'cifar10' or 'cifar100'
        batch_size: Batch size for data loaders
        num_workers: Number of workers for data loading
        sample_size: Fraction of training data to use (0.0 to 1.0)
        val_split: Fraction of training data to use for validation
        random_seed: Random seed for reproducibility
    """
    # Import here to avoid circular imports
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, SubsetRandomSampler
    
    # Set up data directory
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Set random seeds for reproducibility
    torch.manual_seed(random_seed)
    rng = np.random.default_rng(random_seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    
    # Define data transformations
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Load the dataset
    if dataset == 'cifar10':
        train_data = datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=transform_train
        )
        test_data = datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=transform_test
        )
        num_classes = 10
    elif dataset == 'cifar100':
        train_data = datasets.CIFAR100(
            root=data_dir, train=True, download=True, transform=transform_train
        )
        test_data = datasets.CIFAR100(
            root=data_dir, train=False, download=True, transform=transform_test
        )
        num_classes = 100
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
        
    print(f"Dataset loaded from: {data_dir}")
    
    # Split training data into training and validation sets
    num_train = len(train_data)
    indices = np.arange(num_train)
    split = int(np.floor(val_split * num_train))
    
    # Shuffle indices using the new RNG
    rng.shuffle(indices)
    train_idx, val_idx = indices[split:], indices[:split]
    
    # If sample_size is specified, use only a subset of the training data
    if sample_size is not None and sample_size < 1.0:
        sample_size = int(len(train_idx) * sample_size)
        train_idx = train_idx[:sample_size]
    
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    
    # Create data loaders
    train_loader = DataLoader(
        train_data, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        train_data, batch_size=batch_size, sampler=val_sampler,
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, num_classes


def get_data_loaders_simple(dataset='cifar10', batch_size=128, num_workers=4, sample_size=None, val_split=0.1, random_seed=42):
    """A simplified version that only returns train and val loaders.
    
    This is a direct implementation that avoids circular imports.
    """
    # Import here to avoid circular imports
    from torch.utils.data import DataLoader, random_split, Subset
    from torchvision import datasets, transforms
    import numpy as np
    
    # Set random seeds for reproducibility
    torch.manual_seed(random_seed)
    rng = np.random.default_rng(random_seed)
    
    # Define data transformations
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Load dataset
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    if dataset == 'cifar10':
        train_data = datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=transform_train)
    elif dataset == 'cifar100':
        train_data = datasets.CIFAR100(
            root=data_dir, train=True, download=True, transform=transform_train)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    
    # Split training data into training and validation sets
    num_train = len(train_data)
    indices = np.arange(num_train)
    split = int(np.floor(val_split * num_train))
    
    # Shuffle indices using the new RNG
    rng.shuffle(indices)
    train_idx, val_idx = indices[split:], indices[:split]
    
    # If sample_size is specified, use only a subset of the training data
    if sample_size is not None and sample_size < 1.0:
        sample_size = int(len(train_idx) * sample_size)
        train_idx = train_idx[:sample_size]
    
    # Create data loaders
    train_loader = DataLoader(
        Subset(train_data, train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        Subset(train_data, val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def get_data_loaders(dataset='cifar10', batch_size=128, num_workers=4, sample_size=None, val_split=0.1, random_seed=42):
    """Get data loaders for training, validation, and test sets.
    
    Args:
        dataset (str): Dataset name ('cifar10' or 'cifar100')
        batch_size (int): Batch size for data loaders
        num_workers (int): Number of worker processes for data loading
        sample_size (float, optional): Fraction of training data to use (for small-data experiments)
        val_split (float): Fraction of training data to use for validation
        random_seed (int): Random seed for reproducibility
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, num_classes)
    """
    # Get train and val loaders from the simple version
    train_loader, val_loader = get_data_loaders_simple(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sample_size=sample_size,
        val_split=val_split,
        random_seed=random_seed
    )
    
    # Load test data
    from torchvision import datasets, transforms
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    
    if dataset == 'cifar10':
        test_data = datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=transform_test)
        num_classes = 10
    elif dataset == 'cifar100':
        test_data = datasets.CIFAR100(
            root=data_dir, train=False, download=True, transform=transform_test)
        num_classes = 100
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, num_classes


# Alias for backward compatibility
def get_data_loaders(*args, **kwargs):
    """Wrapper to maintain backward compatibility."""
    return get_data_loaders_simple(*args, **kwargs)
