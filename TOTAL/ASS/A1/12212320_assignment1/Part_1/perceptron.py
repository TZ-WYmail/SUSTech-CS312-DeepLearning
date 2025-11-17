import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def make_random_gaussian_2d(
        n_per_class=100,
        test_size=0.2,
        seed=400,
        mu1=None,
        Sig1=None,
        mu2=None,
        Sig2=None,
        return_params=False):
    
    rng = np.random.default_rng(seed)

    # --- 均值 ---
    mu1 = np.asarray(mu1) if mu1 is not None else rng.uniform(-4, 4, size=2)
    mu2 = np.asarray(mu2) if mu2 is not None else rng.uniform(-4, 4, size=2)

    # --- 协方差 ---
    def _rand_cov():
        eig = rng.uniform(0.3, 1.5, size=2)
        theta = rng.uniform(0, 2*np.pi)
        rot = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta),  np.cos(theta)]])
        return rot @ np.diag(eig) @ rot.T

    Sig1 = np.asarray(Sig1) if Sig1 is not None else _rand_cov()
    Sig2 = np.asarray(Sig2) if Sig2 is not None else _rand_cov()
    print("Gaussian 1: 均值 =", mu1, ", 协方差=\n", Sig1)
    print("Gaussian 2: 均值 =", mu2, ", 协方差=\n", Sig2)

    # --- 采样 ---
    X1 = rng.multivariate_normal(mu1, Sig1, size=n_per_class)
    X2 = rng.multivariate_normal(mu2, Sig2, size=n_per_class)
    y1 = np.full(n_per_class, -1)
    y2 = np.full(n_per_class,  1)

    X = np.vstack((X1, X2))
    y = np.hstack((y1, y2))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y)

    if return_params:
        return (X_train, X_test, y_train, y_test), (mu1, Sig1, mu2, Sig2)
    return X_train, X_test, y_train, y_test

def plot_2d_split(X, y, model, title="", h=0.02):
    """画散点 + 决策边界"""
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    mesh = np.c_[xx.ravel(), yy.ravel()]
    Z = model.forward(mesh)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.25, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, cmap='coolwarm', edgecolors='k')
    plt.title(title)
    plt.axis('equal')

def experiment(name, mu1, mu2, Sig1, Sig2, n=200, seed=42, lr=0.01, max_ep=100):
    """训练 + 绘图 + 打印准确率"""
    X_tr, X_te, y_tr, y_te = make_random_gaussian_2d(n_per_class=100, seed=seed,
                                mu1=mu1, mu2=mu2, Sig1=Sig1, Sig2=Sig2,
                                return_params=False)

    model = Perceptron(n_inputs=2, max_epochs=max_ep, learning_rate=lr)
    model.train(X_tr, y_tr)
    acc = np.mean(model.forward(X_te) == y_te)

    print(f"{name}  测试准确率:{acc:.1%}  迭代:{max_ep} 次")

    # ------ 画图 ------
    plt.figure(figsize=(5, 4))
    plot_2d_split(X_tr, y_tr, model, title=name)
    plt.show()

class Perceptron(object):

    def __init__(self, n_inputs, max_epochs=100, learning_rate=0.01):
        self.n_inputs = n_inputs  # Fill in: Initialize number of inputs
        self.max_epochs = max_epochs  # Fill in: Initialize maximum number of epochs
        self.learning_rate = learning_rate  # Fill in: Initialize learning rate
        self.weights = np.zeros(n_inputs + 1)  # Fill in: Initialize weights with zeros (including bias)(最后一项为偏执项)
    def print_info(self):
        print(f"n_inputs: {self.n_inputs}, max_epochs: {self.max_epochs}, learning_rate: {self.learning_rate}, weights: {self.weights}")
            
        
    def forward(self, input_vec):
       """
        Predicts label from input.
        f(x) = sign(w·x + b)
        Args:
            input_vec (np.ndarray): Input array of training data, input vec must be all samples
        Returns:
            int: Predicted label (1 or -1) or Predicted lables.
        """
       dot_product = input_vec @ self.weights[:-1] + self.weights[-1]  # (n,)
       return np.where(dot_product >= 0, 1, -1)       # 批量预测
        
    def train(self, training_inputs, labels):
        for i in range(self.max_epochs): 
            predictions = self.forward(training_inputs)  # (n,)
            miss_indices = (predictions != labels)  # (n,)
            accuracy = np.mean(predictions == labels)
            # 计算权重更新
            weight_gradient = -(labels[miss_indices] @ training_inputs[miss_indices])
            bias_gradient = -np.sum(labels[miss_indices])
            # 更新权重和偏置
            self.weights[:-1] -= self.learning_rate * weight_gradient
            self.weights[-1]  -= self.learning_rate * bias_gradient
            if i < 10:
                print(f"Epoch {i}: Accuracy = {accuracy*100:.2f}%, Misclassified samples = {np.sum(miss_indices)}")
                print("weight_gradient:", weight_gradient, "bias_gradient:", bias_gradient)
                print("Updated weights:", self.weights)

if __name__ == "__main__":
    # 1. 正常情况实验
    experiment("Normal", mu1=[-3, 0], mu2=[3, 0], Sig1=np.eye(2), Sig2=np.eye(2))
    
    # 2. 均值过近实验
    experiment("Close Means", mu1=[-0.5, 0], mu2=[0.5, 0], Sig1=np.eye(2), Sig2=np.eye(2))
    
    # 3. 高方差实验
    experiment("High Variance", mu1=[-3, 0], mu2=[3, 0], Sig1=3*np.eye(2), Sig2=3*np.eye(2))
    
    # 4. 均值过近且高方差实验
    experiment("Close Means & High Variance", mu1=[-0.5, 0], mu2=[0.5, 0], Sig1=3*np.eye(2), Sig2=3*np.eye(2))
