import numpy as np

class Linear(object):
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        self.cache = None
        self.params = {'weight': None, 'bias': None}
        self.params['weight'] = np.random.normal(loc=0.0, scale = 0.5, size = (in_features,out_features))
        self.params['bias'] = np.zeros((1,out_features))
        self.grads = {'weight': None, 'bias': None}

    def forward(self, x):
        print('Linear forward:', x.shape, self.params['weight'].shape, self.params['bias'].shape)
        self.cache = x
        output = x @ self.params['weight'] + self.params['bias']
        return output

    def backward(self, dout):
        x = self.cache
        self.grads['weight'] = x.T @ dout
        self.grads['bias'] = np.sum(dout, axis=0)
        dx = dout @ self.params['weight'].T
        return dx
        
class ReLU(object):
    def __init__(self):
        self.cache = None

    def forward(self, x):
        self.cache = x
        return np.maximum(0, x)

    def backward(self, dout):
        dx = dout.copy()
        dx[self.cache <= 0] = 0
        return dx

class SoftMax(object):
    def __init__(self):
        self.cache = None
        
    def forward(self, x):
        shifted_x = x - np.max(x, axis=1, keepdims=True)  # 数值稳定性
        exp_x = np.exp(shifted_x)
        probs = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        self.cache = probs
        return probs

    def backward(self, dout):
        return dout

class CrossEntropy(object):
    def __init__(self):
        self.cache = None

    def forward(self, x, y):
        N = x.shape[0]
        # --- 手动实现数值稳定性 ---
        max_x = np.max(x, axis=1)
        max_x_reshaped = max_x.reshape(-1, 1)
        exp_x = np.exp(x - max_x_reshaped)
        sum_exp_x = np.sum(exp_x, axis=1)
        sum_exp_x_reshaped = sum_exp_x.reshape(-1, 1)
        probs = exp_x / sum_exp_x_reshaped
        self.cache = probs
        
        # 计算交叉熵损失
        log_probs = np.log(probs[range(N), y] + 1e-12)  # 添加小量防止log(0)
        loss = -np.mean(log_probs)
        
        return loss

    def backward(self, x, y):
        N = x.shape[0]
        probs = self.cache
        dx = probs.copy()
        dx[range(N), y] -= 1
        dx /= N
        return dx