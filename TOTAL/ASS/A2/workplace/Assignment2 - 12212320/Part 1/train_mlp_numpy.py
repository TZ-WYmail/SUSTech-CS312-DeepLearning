# train_mlp_numpy.py

import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from mlp_numpy import MLP
# 从 modules.py 导入所有必要的组件
from modules import *
# --- 默认参数配置 ---
DNN_HIDDEN_UNITS_DEFAULT = "64,32"
LEARNING_RATE_DEFAULT = 0.01
MAX_EPOCHS_DEFAULT = 200
EVAL_FREQ_DEFAULT = 20
OPTIMIZER_DEFAULT = 'sgd' # 新增默认优化器

def generate_data(num):
    data, label = make_moons(n_samples=num, shuffle=True, noise=0.1, random_state=42)
    
    split_idx = int(num * 0.8)
    
    train_X = data[:split_idx]
    train_y = label[:split_idx].astype(int)
    
    test_X = data[split_idx:]
    test_y = label[split_idx:].astype(int)
    
    return train_X, train_y, test_X, test_y

def train_with_data_numpy(X_train,y_train,X_test,y_test,model,dnn_hidden_units, learning_rate, max_steps, eval_freq, optimizer='sgd', batch_size=32):
    loss_fn = CrossEntropy()
    n_train = X_train.shape[0]
    train_losses = []
    test_losses = []             # 新增：记录测试集 loss
    train_accuracies = []
    test_accuracies = []
    for layer in model.layers:
        if layer.__class__.__name__ == 'Linear':
            print(layer.__class__.__name__, layer.in_features, layer.out_features)
        else:
            print(layer.__class__.__name__)
    
    # 用于BGD的梯度累积
    if optimizer == 'bgd':
        accumulated_grads_w = [np.zeros_like(l.params['weight']) for l in model.layers if hasattr(l, 'params')]
        accumulated_grads_b = [np.zeros_like(l.params['bias']) for l in model.layers if hasattr(l, 'params')]

    print(f"Starting training with Optimizer: {optimizer.upper()}, Initial LR: {learning_rate}, Max Epochs: {max_steps}")

    # --- 4. 训练循环 ---
    for epoch in range(max_steps):
        # --- 训练阶段 ---
        if optimizer == 'sgd':

            permutation = np.random.permutation(n_train)
            X_train_shuffled = X_train[permutation]
            y_train_shuffled = y_train[permutation]
            
            for i in range(0, n_train, batch_size):
                X_batch = X_train_shuffled[i:i+batch_size]
                y_batch = y_train_shuffled[i:i+batch_size]
                logits = model.forward(X_batch)
                loss = loss_fn.forward(logits, y_batch)
                dlogits = loss_fn.backward(logits, y_batch)
                model.backward(dlogits)
                for layer in model.layers:
                    if hasattr(layer, 'params'):
                        layer.params['weight'] -= learning_rate * layer.grads['weight']
                        layer.params['bias'] -= learning_rate * layer.grads['bias']

        elif optimizer == 'bgd':
            logits = model.forward(X_train)
            loss = loss_fn.forward(logits, y_train)
            dlogits = loss_fn.backward(logits, y_train)
            model.backward(dlogits)
            
            # 累积梯度
            grad_idx = 0
            for layer in model.layers:
                if hasattr(layer, 'params'):
                    accumulated_grads_w[grad_idx] += layer.grads['weight']
                    accumulated_grads_b[grad_idx] += layer.grads['bias']
                    grad_idx += 1
            
            # 参数更新
            grad_idx = 0
            for layer in model.layers:
                if hasattr(layer, 'params'):
                    layer.params['weight'] -= learning_rate * accumulated_grads_w[grad_idx]
                    layer.params['bias'] -= learning_rate * accumulated_grads_b[grad_idx]
                    grad_idx += 1
            
            # 清空累积的梯度
            for i in range(len(accumulated_grads_w)):
                accumulated_grads_w[i].fill(0)
                accumulated_grads_b[i].fill(0)

        # --- 评估阶段 ---
        if (epoch + 1) % eval_freq == 0:
            # 训练集 logits & loss & 准确率（全量）
            train_logits = model.forward(X_train)
            train_preds = np.argmax(train_logits, axis=1)
            train_true = y_train
            train_accuracy = np.mean(train_preds == train_true)
            train_loss = loss_fn.forward(train_logits, train_true)   # 计算训练集 loss（标量）

            # 测试集 logits & loss & 准确率（全量）
            test_logits = model.forward(X_test)
            test_preds = np.argmax(test_logits, axis=1)
            test_true = y_test
            test_accuracy = np.mean(test_preds == test_true)
            test_loss = loss_fn.forward(test_logits, test_true)      # 计算测试集 loss（标量）
            
            # 记录指标
            train_losses.append(train_loss)
            test_losses.append(test_loss)       # 新增：记录 test loss
            train_accuracies.append(train_accuracy)
            test_accuracies.append(test_accuracy)
            
            # 打印进度（包含 test loss）
            print(f"Epoch {epoch+1}/{max_steps}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Train Acc: {train_accuracy:.4f}, Test Acc: {test_accuracy:.4f}")
    
    # --- 6. 打印最终测试准确率 ---
    final_test_acc = test_accuracies[-1] if test_accuracies else 0
    print(f"\nFinal Test Accuracy with {optimizer.upper()}: {final_test_acc:.4f}")
    
    # --- 新增/修改：绘制训练和测试的 loss 和 accuracy 曲线 ---
    import matplotlib.pyplot as plt
    
    epochs = np.arange(eval_freq, max_steps + 1, eval_freq)  # 评估的 epoch 点
    
    plt.figure(figsize=(12, 5))
    
    # 子图1: Loss 曲线（同时绘制 train_loss 与 test_loss）
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, test_losses, label='Test Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training/Test Loss Curve')
    plt.legend()
    plt.grid(True)
    
    # 子图2: Accuracy 曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy', marker='o')
    plt.plot(epochs, test_accuracies, label='Test Accuracy', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curves')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
def train(dnn_hidden_units, learning_rate, max_steps, eval_freq, optimizer='sgd', batch_size=32):
    """
    训练和评估模型
    
    Args:
        dnn_hidden_units (str): 逗号分隔的隐藏层单元数列表，例如 "64,32"
        learning_rate (float): 初始学习率
        max_steps (int): 训练轮数
        eval_freq (int): 每隔多少轮评估一次
        optimizer (str): 'sgd' (随机梯度下降) 或 'bgd' (批量梯度下降)
        batch_size (int): 批量大小，仅用于SGD
    """
    # --- 1. 数据准备 ---
    # generate_data already returns train and test splits (train_X, train_y, test_X, test_y)
    X_train, y_train, X_test, y_test = generate_data(1000)

    # --- 2. 模型构建 ---
    hidden_units_list = [int(unit) for unit in dnn_hidden_units.split(',')]
    model = MLP(n_inputs=2, n_hidden=hidden_units_list, n_classes=2)
    loss_fn = CrossEntropy()
    # 注意：CrossEntropy 会在内部计算 softmax，因此无需在模型末尾添加 SoftMax 层

    # --- 3. 训练设置 ---
    n_train = X_train.shape[0]
    # 使用传入的 batch_size 参数（默认 32），不要在此处覆盖它
    train_losses = []
    test_losses = []             # 新增：记录测试集 loss
    train_accuracies = []
    test_accuracies = []
    
    # 用于BGD的梯度累积
    if optimizer == 'bgd':
        accumulated_grads_w = [np.zeros_like(l.params['weight']) for l in model.layers if hasattr(l, 'params')]
        accumulated_grads_b = [np.zeros_like(l.params['bias']) for l in model.layers if hasattr(l, 'params')]

    print(f"Starting training with Optimizer: {optimizer.upper()}, Initial LR: {learning_rate}, Max Epochs: {max_steps}")

    # --- 4. 训练循环 ---
    for epoch in range(max_steps):
        # --- 训练阶段 ---
        if optimizer == 'sgd':

            permutation = np.random.permutation(n_train)
            X_train_shuffled = X_train[permutation]
            y_train_shuffled = y_train[permutation]
            
            for i in range(0, n_train, batch_size):
                X_batch = X_train_shuffled[i:i+batch_size]
                y_batch = y_train_shuffled[i:i+batch_size]
                logits = model.forward(X_batch)
                loss = loss_fn.forward(logits, y_batch)
                dlogits = loss_fn.backward(logits, y_batch)
                model.backward(dlogits)
                for layer in model.layers:
                    if hasattr(layer, 'params'):
                        layer.params['weight'] -= learning_rate * layer.grads['weight']
                        layer.params['bias'] -= learning_rate * layer.grads['bias']

        elif optimizer == 'bgd':
            logits = model.forward(X_train)
            loss = loss_fn.forward(logits, y_train)
            dlogits = loss_fn.backward(logits, y_train)
            model.backward(dlogits)
            
            # 累积梯度
            grad_idx = 0
            for layer in model.layers:
                if hasattr(layer, 'params'):
                    accumulated_grads_w[grad_idx] += layer.grads['weight']
                    accumulated_grads_b[grad_idx] += layer.grads['bias']
                    grad_idx += 1
            
            # 参数更新
            grad_idx = 0
            for layer in model.layers:
                if hasattr(layer, 'params'):
                    layer.params['weight'] -= learning_rate * accumulated_grads_w[grad_idx]
                    layer.params['bias'] -= learning_rate * accumulated_grads_b[grad_idx]
                    grad_idx += 1
            
            # 清空累积的梯度
            for i in range(len(accumulated_grads_w)):
                accumulated_grads_w[i].fill(0)
                accumulated_grads_b[i].fill(0)

        # --- 评估阶段 ---
        if (epoch + 1) % eval_freq == 0:
            # 训练集 logits & loss & 准确率（全量）
            train_logits = model.forward(X_train)
            train_preds = np.argmax(train_logits, axis=1)
            train_true = y_train
            train_accuracy = np.mean(train_preds == train_true)
            train_loss = loss_fn.forward(train_logits, train_true)   # 计算训练集 loss（标量）

            # 测试集 logits & loss & 准确率（全量）
            test_logits = model.forward(X_test)
            test_preds = np.argmax(test_logits, axis=1)
            test_true = y_test
            test_accuracy = np.mean(test_preds == test_true)
            test_loss = loss_fn.forward(test_logits, test_true)      # 计算测试集 loss（标量）
            
            # 记录指标
            train_losses.append(train_loss)
            test_losses.append(test_loss)       # 新增：记录 test loss
            train_accuracies.append(train_accuracy)
            test_accuracies.append(test_accuracy)
            
            # 打印进度（包含 test loss）
            print(f"Epoch {epoch+1}/{max_steps}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Train Acc: {train_accuracy:.4f}, Test Acc: {test_accuracy:.4f}")

    # --- 5. 绘制结果 ---
    plot_metrics(train_losses, train_accuracies, test_accuracies)
    
    # --- 6. 打印最终测试准确率 ---
    final_test_acc = test_accuracies[-1] if test_accuracies else 0
    print(f"\nFinal Test Accuracy with {optimizer.upper()}: {final_test_acc:.4f}")

def plot_metrics(train_losses, train_accuracies, test_accuracies):
    """绘制训练过程中的损失和准确率曲线"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Evaluation Step')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Evaluation Step')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function.
    """
    # Parsing command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_EPOCHS_DEFAULT,
                        help='Number of epochs to run trainer')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    # --- 新增优化器参数 ---
    parser.add_argument('--optimizer', type=str, default=OPTIMIZER_DEFAULT, choices=['sgd', 'bgd'],
                        help='Optimizer to use: sgd or bgd')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size to use for SGD (only used when --optimizer sgd)')
    
    FLAGS = parser.parse_known_args()[0]
    
    # 调用训练函数，并传入所有参数，包括新的optimizer
    train(FLAGS.dnn_hidden_units, FLAGS.learning_rate, FLAGS.max_steps, FLAGS.eval_freq, FLAGS.optimizer, FLAGS.batch_size)

if __name__ == '__main__':
    main()
