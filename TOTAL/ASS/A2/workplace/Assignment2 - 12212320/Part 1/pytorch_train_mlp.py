import argparse
import numpy as np
import os
from pytorch_mlp import MLP
import torch
import torch.nn as nn
from torch.optim import SGD
import matplotlib.pyplot as plt  # 新增：用于绘图

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '20'
LEARNING_RATE_DEFAULT = 1e-1
MAX_EPOCHS_DEFAULT = 1500
EVAL_FREQ_DEFAULT = 10

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
    preds = np.argmax(predictions, axis=1)
    labels = np.argmax(targets, axis=1)
    return np.mean(preds == labels)


from torch.utils.data import TensorDataset, DataLoader

def train_with_data_pytorch(X_train, y_train, X_test, y_test, model,
                            learning_rate=1e-2, max_steps=1000, eval_freq=50,
                            batch_size=32):
    """
    Args:
        X_train, y_train, X_test, y_test: numpy arrays (as in notebook)
        model: an instance of pytorch_mlp.MLP
        batch_size: mini-batch size (default 32)
    """
    # ---- 数据预处理（与原函数一致） ----
    X_train = np.asarray(X_train, dtype=np.float32)
    X_test  = np.asarray(X_test,  dtype=np.float32)
    if y_train.ndim > 1:
        y_train = np.argmax(y_train, axis=1)
    if y_test.ndim > 1:
        y_test = np.argmax(y_test, axis=1)
    y_train = np.asarray(y_train, dtype=np.int64)
    y_test  = np.asarray(y_test,  dtype=np.int64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=learning_rate)

    # ---- 构建 DataLoader ----
    train_dataset = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).long()
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 测试集张量一次性搬到设备
    X_test_t = torch.from_numpy(X_test).float().to(device)
    y_test_t = torch.from_numpy(y_test).long().to(device)
    # 新增：训练集整体验证用张量
    X_train_t = torch.from_numpy(X_train).float().to(device)
    y_train_t = torch.from_numpy(y_train).long().to(device)

    # 用于记录曲线的数据
    steps_list = []
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []

    step_cnt = 0
    while step_cnt < max_steps:
        for xb, yb in train_loader:
            if step_cnt >= max_steps:
                break

            xb, yb = xb.to(device), yb.to(device)
            model.train()
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            step_cnt += 1

            # ---- 评估 ----
            if step_cnt % eval_freq == 0 or step_cnt == max_steps:
                model.eval()
                with torch.no_grad():
                    # 训练集（全量） - 计算 loss 和 acc
                    out_train = model(X_train_t)
                    loss_train = criterion(out_train, y_train_t).item()
                    acc_train = (out_train.argmax(1) == y_train_t).float().mean().item()

                    # 测试集 - 计算 loss 和 acc
                    out_test = model(X_test_t)
                    loss_test = criterion(out_test, y_test_t).item()
                    acc_test = (out_test.argmax(1) == y_test_t).float().mean().item()

                # 记录
                steps_list.append(step_cnt)
                train_losses.append(loss_train)
                test_losses.append(loss_test)
                train_accs.append(acc_train)
                test_accs.append(acc_test)

                print(f"Step {step_cnt}/{max_steps} - batch_loss: {loss.item():.4f} - "
                      f"train_loss: {loss_train:.4f} - test_loss: {loss_test:.4f} - "
                      f"train_acc: {acc_train:.4f} - test_acc: {acc_test:.4f}")

    # 训练结束，绘图并保存
        # 训练结束，直接显示曲线
    if len(steps_list) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # 左图：训练/测试 loss
        ax1.plot(steps_list, train_losses, label='train_loss', marker='o')
        ax1.plot(steps_list, test_losses, label='test_loss', marker='s')
        ax1.set_xlabel('step'); ax1.set_ylabel('loss')
        ax1.set_title('Loss Curve')
        ax1.legend(); ax1.grid(True)

        # 右图：训练/测试 accuracy
        ax2.plot(steps_list, train_accs, label='train_acc', marker='o')
        ax2.plot(steps_list, test_accs, label='test_acc', marker='s')
        ax2.set_xlabel('step'); ax2.set_ylabel('accuracy')
        ax2.set_title('Accuracy Curve')
        ax2.legend(); ax2.grid(True)

        plt.tight_layout()
        plt.show()
    else:
        print("No evaluation steps were recorded (check eval_freq and max_steps).")

    return model


def train(dnn_hidden_units, learning_rate, max_steps, eval_freq, optimizer='sgd', batch_size=32):
    """
    Performs training and evaluation of MLP model.
    NOTE: You should the model on the whole test set each eval_freq iterations.
    """
    # Parse hidden units
    hidden_units = [int(u) for u in dnn_hidden_units.split(',')]

    # Generate make_moons dataset
    from sklearn.datasets import make_moons
    from sklearn.model_selection import train_test_split
    X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
    y_onehot = np.eye(2)[y]  # one-hot encode

    X_train, X_test, y_train, y_test = train_test_split(X, y_onehot,
                                                        test_size=0.2,
                                                        random_state=42)

    # Convert to float32
    X_train = X_train.astype(np.float32)
    X_test  = X_test.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_test  = y_test.astype(np.float32)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(n_inputs=2, n_hidden=hidden_units, n_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_fn = SGD(model.parameters(), lr=learning_rate)

    # 新增：用于记录指标
    steps_list = []
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []

    # 转换为 tensor 并移到 device
    X_train_t = torch.from_numpy(X_train).to(device)
    y_train_t = torch.from_numpy(y_train).to(device)
    X_test_t = torch.from_numpy(X_test).to(device)
    y_test_t = torch.from_numpy(y_test).to(device)

    # Training loop
    for step in range(max_steps):
        # Forward pass
        model.train()
        logits = model(X_train_t)
        y_train_labels = y_train_t.max(dim=1)[1]  # class indices
        loss = criterion(logits, y_train_labels)

        # Backward pass
        optimizer_fn.zero_grad()
        loss.backward()
        optimizer_fn.step()

        # Evaluation
        if step % eval_freq == 0 or step == max_steps - 1:
            model.eval()
            with torch.no_grad():
                # 训练集指标
                train_logits = model(X_train_t)
                train_loss = criterion(train_logits, y_train_labels).item()
                train_acc = (train_logits.argmax(1) == y_train_labels).float().mean().item()

                # 测试集指标
                test_logits = model(X_test_t)
                y_test_labels = y_test_t.max(dim=1)[1]
                test_loss = criterion(test_logits, y_test_labels).item()
                test_acc = (test_logits.argmax(1) == y_test_labels).float().mean().item()

                # 记录
                steps_list.append(step)
                train_losses.append(train_loss)
                test_losses.append(test_loss)
                train_accs.append(train_acc)
                test_accs.append(test_acc)

                print(f"step {step:5d} | train_loss {train_loss:.4f} train_acc {train_acc:.4f} | "
                      f"test_loss {test_loss:.4f} test_acc {test_acc:.4f}")

    # 训练结束，绘制曲线
    if len(steps_list) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # 左图：训练/测试 loss
        ax1.plot(steps_list, train_losses, label='train_loss', marker='o')
        ax1.plot(steps_list, test_losses, label='test_loss', marker='s')
        ax1.set_xlabel('step'); ax1.set_ylabel('loss')
        ax1.set_title('Loss Curve')
        ax1.legend(); ax1.grid(True)

        # 右图：训练/测试 accuracy
        ax2.plot(steps_list, train_accs, label='train_acc', marker='o')
        ax2.plot(steps_list, test_accs, label='test_acc', marker='s')
        ax2.set_xlabel('step'); ax2.set_ylabel('accuracy')
        ax2.set_title('Accuracy Curve')
        ax2.legend(); ax2.grid(True)

        plt.tight_layout()
        plt.show()
    else:
        print("No evaluation steps were recorded.")

def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
    parser.add_argument('--max_steps', type = int, default = MAX_EPOCHS_DEFAULT,
                      help='Number of epochs to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                          help='Frequency of evaluation on the test set')
    FLAGS, unparsed = parser.parse_known_args()
    
    train(FLAGS.dnn_hidden_units, FLAGS.learning_rate, FLAGS.max_steps, FLAGS.eval_freq)

if __name__ == '__main__':
    # Command line arguments
    main()