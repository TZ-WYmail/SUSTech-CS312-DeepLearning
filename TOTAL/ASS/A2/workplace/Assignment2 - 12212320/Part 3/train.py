from __future__ import absolute_import, division, print_function
import argparse, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import PalindromeDataset          # 你的代码
from vanilla_rnn import VanillaRNN             # 你即将实现

def train(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VanillaRNN(input_length=config.input_length,
                       input_dim=config.input_dim, 
                       hidden_dim=config.num_hidden,
                       output_dim=config.num_classes,
                       batch_size=config.batch_size).to(device)
    dataset = PalindromeDataset(config.input_length)
    loader  = DataLoader(dataset, batch_size=config.batch_size,
                         shuffle=True, num_workers=0)
    # 3. 损失 & 优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate)
    # 4. 记录
    losses, accs = [], []
    # 5. 主循环
    step_iter = tqdm(range(config.train_steps), desc='Steps')
    for step, (seq, label) in zip(step_iter, loader):
        seq   = seq[:, :config.input_length].to(device)        
        label = label.to(device)
        # 前向
        logits = model(seq)            # [B, 10]
        loss   = criterion(logits, label)
        # 反向
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.max_norm)
        optimizer.step()
        # 记录
        acc = (logits.argmax(1) == label).float().mean().item()
        losses.append(loss.item())
        accs.append(acc)
        if (step + 1) % 20 == 0:
            step_iter.set_postfix(loss=f'{loss.item():.4f}', acc=f'{acc:.4f}')

        if step + 1 >= config.train_steps:
            break

    print('Done training.')
    return model, {'loss': losses, 'acc': accs}


# 命令行入口保留
if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--input_length', type=int, default=10, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)

    config = parser.parse_args()
    # Train the model
    train(config)