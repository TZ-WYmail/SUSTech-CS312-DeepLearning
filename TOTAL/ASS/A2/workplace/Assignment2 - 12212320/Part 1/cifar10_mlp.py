from __future__ import absolute_import, division, print_function
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import os, argparse
from tqdm import tqdm  # 新增：训练进度条

# ---------- 1. 网络：仅 Linear + ReLU + Dropout ----------
class MLP(nn.Module):
    def __init__(self, hidden_nodes=2048, dropout=0.5, n_layers=3):
        super().__init__()
        layers = [nn.Flatten()] 
        in_dim = 3072
        for i in range(n_layers):
            layers += [nn.Linear(in_dim, hidden_nodes),
                       nn.ReLU(inplace=True),
                       nn.Dropout(dropout)]
            in_dim = hidden_nodes
        layers += [nn.Linear(in_dim, 10)] 
        self.net = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


def get_loaders(batch_size=128, root='./data'):
    mean = [0.4914, 0.4822, 0.4465]
    std  = [0.2470, 0.2435, 0.2616]
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize(mean, std)])
    train_set = datasets.CIFAR10(root=root, train=True,
                                 download=True, transform=tf)
    test_set  = datasets.CIFAR10(root=root, train=False,
                                 transform=tf)
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader


def train_mlp(hidden_nodes=1024, dropout=0.5, n_layers=3,
              lr=0.1, batch_size=128, epochs=100, eval_freq=5,
              optimizer='SGD', root='./data', device=None, plot=True):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Using device:", device)

    train_loader, test_loader = get_loaders(batch_size, root)
    model = MLP(hidden_nodes, dropout, n_layers).to(device)

    # 添加：检查模型初始权重是否合理
    print("\n=== 检查模型初始化 ===")
    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"{name}: mean={param.data.mean().item():.4f}, std={param.data.std().item():.4f}, max={param.data.abs().max().item():.4f}")

    criterion = nn.CrossEntropyLoss()
    if optimizer.upper() == 'SGD':
        opt = torch.optim.SGD(model.parameters(), lr=lr,
                              momentum=0.9, weight_decay=5e-4)
    elif optimizer.upper() == 'ADAM':
        opt = torch.optim.Adam(model.parameters(), lr=lr,
                               weight_decay=5e-4)
    else:
        raise ValueError('optimizer must be SGD or ADAM')

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    train_acc_curve, test_acc_curve = [], []
    train_loss_curve, test_loss_curve = [], []
    x_axis = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0., 0, 0
        
        # 添加：epoch 级别的梯度统计
        epoch_grad_norm = 0.0
        batch_count = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{epochs}', leave=False)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            
            if torch.isnan(loss):
                print(f"\n!!! NaN detected at epoch {epoch}, batch {batch_count}")
                print(f"Output stats: min={out.min().item():.4f}, max={out.max().item():.4f}, mean={out.mean().item():.4f}")
                print(f"Input stats: min={x.min().item():.4f}, max={x.max().item():.4f}")
                # 打印最近几层的权重
                for name, param in model.named_parameters():
                    if 'weight' in name:
                        print(f"{name}: has_nan={torch.isnan(param).any().item()}, max={param.abs().max().item():.4f}")
                raise ValueError("Training diverged with NaN loss")
            
            loss.backward()
            
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            epoch_grad_norm += grad_norm.item()
            
            opt.step()

            batch_loss = loss.item()
            batch_acc = (out.argmax(1) == y).float().mean().item()
            running_loss += batch_loss * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)
            batch_count += 1
            
            pbar.set_postfix({'batch_loss': f'{batch_loss:.4f}', 
                            'batch_acc': f'{batch_acc:.4f}',
                            'grad_norm': f'{grad_norm:.2f}'})

        scheduler.step()

        train_acc = correct / total
        train_loss = running_loss / total
        avg_grad_norm = epoch_grad_norm / batch_count

        # 添加：梯度统计输出
        if epoch <= 5 or epoch % eval_freq == 0:
            print(f"\nEpoch {epoch}: avg_grad_norm={avg_grad_norm:.4f}, lr={scheduler.get_last_lr()[0]:.6f}")

        if epoch % eval_freq == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                test_loss, test_correct, test_total = 0., 0, 0
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)
                    out = model(x)
                    loss = criterion(out, y)
                    test_loss += loss.item() * x.size(0)
                    test_correct += (out.argmax(1) == y).sum().item()
                    test_total += x.size(0)

                test_acc = test_correct / test_total
                test_loss /= test_total

                train_acc_curve.append(train_acc)
                train_loss_curve.append(train_loss)
                test_acc_curve.append(test_acc)
                test_loss_curve.append(test_loss)
                x_axis.append(epoch)

                print(f'Epoch {epoch:03d} | '
                      f'train loss {train_loss:.4f} acc {train_acc:.4f} | '
                      f'test loss {test_loss:.4f} acc {test_acc:.4f}')

    
    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6))
        ax1.plot(x_axis, train_acc_curve, 'r-', label='train')
        ax1.plot(x_axis, test_acc_curve,  'b-', label='test')
        ax1.set_ylabel('Accuracy'); ax1.legend()
        ax2.plot(x_axis, train_loss_curve, 'g-', label='train')
        ax2.plot(x_axis, test_loss_curve,  'orange', label='test')
        ax2.set_ylabel('Loss'); ax2.set_xlabel('Epoch'); ax2.legend()
        plt.tight_layout(); plt.show()

    return model, (train_acc_curve, test_acc_curve,
                   train_loss_curve, test_loss_curve)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_nodes', type=int, default=1024)  # 降低默认值
    parser.add_argument('--dropout', type=float, default=0.3)      # 降低 dropout
    parser.add_argument('--n_layers', type=int, default=2)         # 减少层数
    parser.add_argument('--lr', type=float, default=0.01)          # 降低学习率
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--optimizer', type=str, default='SGD')
    parser.add_argument('--eval_freq', type=int, default=5)
    parser.add_argument('--root', type=str, default='./data')
    args = parser.parse_args()

    train_mlp(hidden_nodes=args.hidden_nodes,
              dropout=args.dropout,
              n_layers=args.n_layers,
              lr=args.lr,
              batch_size=args.batch_size,
              epochs=args.epochs,
              optimizer=args.optimizer,
              eval_freq=args.eval_freq,
              root=args.root)


if __name__ == '__main__':
    main()