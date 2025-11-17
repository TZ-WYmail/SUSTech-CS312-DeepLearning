from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
import torch
from tqdm import tqdm
from cnn_model import CNN

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_EPOCHS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'
DATA_DIR_DEFAULT = 'Part 2/data'
FLAGS = None

def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e., the average of correct predictions
    of the network.
    Args:
        predictions: 2D float array of size [number_of_data_samples, n_classes]
        labels: 2D int array of size [number_of_data_samples, n_classes] with one-hot encoding of ground-truth labels
    Returns:
        accuracy: scalar float, the accuracy of predictions.
    """
    targ_labels = np.argmax(targets, axis=1)
    pred_labels = np.argmax(predictions, axis=1)
    return np.mean(pred_labels == targ_labels)

def train_one_epoch(model, train_loader, optimizer, criterion, device):
    """
    训练一个 epoch
    """
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total=0
    for x, y in train_loader:
        x,y=x.to(device),y.to(device)
        optimizer.zero_grad()
        logits=model(x)
        loss=criterion(logits,y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
        running_corrects += (logits.argmax(1) == y).sum().item()
        total+=x.size(0)
    return running_loss /total, running_corrects / total
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, running_correct, total = 0., 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        running_loss += loss.item() * x.size(0)
        running_correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)

    avg_loss = running_loss / total
    avg_acc  = running_correct / total
    return avg_loss, avg_acc

def train(model,train_loader, val_loader, optimizer, criterion, device, max_epochs=150, eval_freq=10):
    """
    训练模型，使用 accuracy 计算准确率
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    model.to(device)
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in tqdm(range(max_epochs), desc='Epoch', position=0, leave=True):
        # 1. 训练
        train_loss, train_acc = train_one_epoch(model, train_loader,
                                                optimizer, criterion, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # 2. 验证
        if (epoch + 1) % eval_freq == 0 or epoch == max_epochs - 1:
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)

            print(f'Epoch {epoch+1}/{max_epochs}, '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    return {'train_loss': train_losses,
            'train_acc':  train_accuracies,
            'val_loss':   val_losses,
            'val_acc':    val_accuracies}

import matplotlib.pyplot as plt
def plot_curve(history):
    plt.figure(figsize=(6, 4))
    plt.plot(history['train_loss'], label='Train Loss', color='tab:red')
    plt.plot(history['val_loss'],   label='Val Loss',  color='tab:blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    # 2. 准确率曲线
    plt.figure(figsize=(6, 4))
    plt.plot(history['train_acc'], label='Train Acc', color='tab:red')
    plt.plot(history['val_acc'],   label='Val Acc',  color='tab:blue')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader

def load_cifar10_vgg(batch_size=128, root='./data', num_workers=2):

    transform = transforms.ToTensor()
    train_val_set = datasets.CIFAR10(root=root, train=True,
                                     download=True, transform=transform)
    test_set = datasets.CIFAR10(root=root, train=False,
                                download=True, transform=transform)
    # 80 % / 20 % 划分
    train_size = int(0.8 * len(train_val_set))
    val_size   = len(train_val_set) - train_size
    train_data, val_data = random_split(
        train_val_set, [train_size, val_size],
        generator=torch.Generator().manual_seed(42))
    # DataLoader
    train_loader = DataLoader(train_data, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers,
                              pin_memory=True)
    val_loader   = DataLoader(val_data,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)
    test_loader  = DataLoader(test_set,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)

    return train_loader, val_loader, test_loader

def main(FLAGS):
    """
    Main function
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model     = CNN(n_channels=3, n_classes=10).to(device)
    train_loader, val_loader, test_loader = load_cifar10_vgg(
        batch_size=FLAGS.batch_size, root=FLAGS.data_dir)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.learning_rate)
    history = train(model, train_loader, val_loader,
                    optimizer, criterion, device,
                    max_epochs=FLAGS.max_steps, eval_freq=FLAGS.eval_freq)
    plot_curve(history)

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_EPOCHS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  FLAGS = parser.parse_known_args()[0]
  main(FLAGS)