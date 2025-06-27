#!/usr/bin/env python3
"""
Minimal CIFAR Training Script
==========================

A simplified training script for CIFAR-10/100 models.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Minimal CIFAR Training')
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100'],
                      help='Dataset name (default: cifar10)')
    parser.add_argument('--data-dir', default='./data', help='Dataset directory')
    parser.add_argument('--data-fraction', type=float, default=1.0,
                      help='Fraction of training data to use (0.0 to 1.0)')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--model', default='resnet18', help='Model architecture')
    parser.add_argument('--optimizer', default='adam', choices=['adam', 'sgd'],
                      help='Optimizer (default: adam)')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to use for training')
    return parser.parse_args()

# Create model
def create_model(model_name, num_classes):
    if model_name == 'resnet18':
        model = torchvision.models.resnet18(num_classes=num_classes)
    elif model_name == 'resnet34':
        model = torchvision.models.resnet34(num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model

# Create data loaders
def get_data_loaders(dataset, data_dir, batch_size, data_fraction=1.0):
    # Data augmentation and normalization
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
    
    # Load datasets
    if dataset == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=transform_test)
        num_classes = 10
    elif dataset == 'cifar100':
        train_dataset = torchvision.datasets.CIFAR100(
            root=data_dir, train=True, download=True, transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR100(
            root=data_dir, train=False, download=True, transform=transform_test)
        num_classes = 100
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    
    # Split training set into train/val (90/10)
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size])
    
    # Apply data fraction if needed
    if data_fraction < 1.0:
        subset_size = int(len(train_dataset) * data_fraction)
        train_dataset = torch.utils.data.Subset(train_dataset, range(subset_size))
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader, num_classes

# Training function
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in tqdm(train_loader, desc='Training', leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update metrics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total
    return train_loss, train_acc

# Validation function
def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc='Validating', leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Update metrics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    val_loss = running_loss / len(val_loader)
    val_acc = 100. * correct / total
    return val_loss, val_acc

# Test function
def test(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc='Testing', leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Update metrics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    test_loss = running_loss / len(test_loader)
    test_acc = 100. * correct / total
    return test_loss, test_acc

def main():
    args = parse_args()
    
    print(f"\nTraining {args.model.upper()} on {args.dataset.upper()} with {args.data_fraction*100}% of data")
    print(f"Using device: {args.device}\n")
    
    # Create data loaders
    train_loader, val_loader, test_loader, num_classes = get_data_loaders(
        args.dataset, args.data_dir, args.batch_size, args.data_fraction)
    
    # Create model
    model = create_model(args.model, num_classes).to(args.device)
    
    # Create loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:  # sgd
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    best_val_acc = 0.0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, args.device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, args.device)
        
        # Update learning rate
        scheduler.step()
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, 'best_model.pth')
    
    # Test the best model
    print("\nTesting best model...")
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_acc = test(model, test_loader, criterion, args.device)
    
    print(f"\nTest Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    print(f"Best Val Acc: {best_val_acc:.2f}%")

if __name__ == '__main__':
    main()
