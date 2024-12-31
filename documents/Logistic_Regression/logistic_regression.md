
## 逻辑回归的数学推导

### 1. 引言

逻辑回归是一种用于二分类问题的监督学习算法。与线性回归不同，逻辑回归旨在预测离散的类别标签（通常是0或1），而不是连续的数值。逻辑回归通过使用**逻辑函数（Sigmoid函数）**将线性组合的输出转换为概率值，从而进行分类决策。

### 2. 假设函数

在逻辑回归中，假设函数定义为：

$$
h_\theta(\mathbf{x}) = \sigma(\mathbf{\theta}^T \mathbf{x}) = \frac{1}{1 + e^{-\mathbf{\theta}^T \mathbf{x}}}
$$

其中：

- $h_\theta(\mathbf{x})$ 是输入 $\mathbf{x}$ 属于类别1的预测概率。
- $\sigma(z) = \frac{1}{1 + e^{-z}}$ 是 Sigmoid 函数。
- $\mathbf{\theta} = [\theta_0, \theta_1, \dots, \theta_m]^T$ 是模型参数。
- $\mathbf{x} = [1, x_1, x_2, \dots, x_m]^T$ 是输入特征向量（包括偏置项 $x_0 = 1$）。

**图示 Sigmoid 函数：**

![Sigmoid Function](https://upload.wikimedia.org/wikipedia/commons/8/88/Logistic-curve.svg)

### 3. 损失函数

逻辑回归使用**对数损失函数（Log Loss）**，也称为**交叉熵损失函数（Cross-Entropy Loss）**，来衡量预测概率与真实标签之间的差距。

对于单个训练样本 $(\mathbf{x}^{(i)}, y^{(i)})$，损失函数定义为：

$$
\text{Loss}(h_\theta(\mathbf{x}^{(i)}), y^{(i)}) = -\left[ y^{(i)} \log(h_\theta(\mathbf{x}^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(\mathbf{x}^{(i)})) \right]
$$

对于整个训练集，损失函数为平均对数损失：

$$
J(\mathbf{\theta}) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(h_\theta(\mathbf{x}^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(\mathbf{x}^{(i)})) \right]
$$

其中：

- $m$ 是训练样本的数量。
- $h_\theta(\mathbf{x}^{(i)})$ 是第 $i$ 个样本的预测概率。
- $y^{(i)}$ 是第 $i$ 个样本的真实标签，取值为0或1。

**为何选择对数损失函数？**

对数损失函数具有以下优点：

1. **凸性**：在逻辑回归中，对数损失函数是凸函数，这保证了优化算法能够找到全局最优解。
2. **概率解释**：对数损失函数直接衡量了模型预测的概率与真实标签之间的差距。
3. **梯度友好**：对数损失函数在梯度下降过程中具有良好的性质，有助于稳定和高效的优化。

### 4. 参数优化

我们的目标是找到一组参数 $\mathbf{\theta}$，使得损失函数 $J(\mathbf{\theta})$ 最小。常用的优化方法包括梯度下降法、牛顿法和拟牛顿法等。这里我们主要讨论梯度下降法的推导过程。

#### 4.1 梯度下降法

梯度下降法是一种迭代优化算法，通过不断沿着损失函数梯度的反方向更新参数，逐步逼近损失函数的最小值。

**参数更新规则：**

$$
\theta_j := \theta_j - \alpha \frac{\partial J(\mathbf{\theta})}{\partial \theta_j}
$$

其中：

- $\alpha$ 是学习率，控制每次更新的步长。
- $\frac{\partial J(\mathbf{\theta})}{\partial \theta_j}$ 是损失函数关于参数 $\theta_j$ 的偏导数。

**计算偏导数：**

首先，计算损失函数 $J(\mathbf{\theta})$ 关于参数 $\theta_j$ 的偏导数：

$$
\frac{\partial J(\mathbf{\theta})}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} \left( h_\theta(\mathbf{x}^{(i)}) - y^{(i)} \right) x_j^{(i)}
$$

因此，参数更新规则可以具体表示为：

$$
\theta_j := \theta_j - \alpha \left( \frac{1}{m} \sum_{i=1}^{m} \left( h_\theta(\mathbf{x}^{(i)}) - y^{(i)} \right) x_j^{(i)} \right)
$$

**向量化表示：**

为了提高计算效率，我们可以将上述更新规则向量化：

$$
\mathbf{\theta} := \mathbf{\theta} - \alpha \cdot \frac{1}{m} \mathbf{X}^T (\mathbf{h} - \mathbf{y})
$$

其中：

- $\mathbf{X}$ 是特征矩阵，形状为 $(m, n+1)$。
- $\mathbf{h}$ 是预测概率向量，形状为 $(m, 1)$。
- $\mathbf{y}$ 是真实标签向量，形状为 $(m, 1)$。

#### 4.2 梯度下降算法步骤

1. **初始化参数**：通常将参数 $\mathbf{\theta}$ 初始化为全零或小的随机值。
2. **计算预测概率**：利用 Sigmoid 函数计算每个样本的预测概率。
3. **计算梯度**：计算损失函数相对于每个参数的偏导数。
4. **更新参数**：根据梯度和学习率更新参数。
5. **迭代**：重复步骤2-4，直到损失函数收敛或达到最大迭代次数。

### 5. 正则化

为了防止模型过拟合，可以在损失函数中添加正则化项。常见的正则化方法有 L1 正则化和 L2 正则化。

#### 5.1 L2 正则化（岭回归）

加入 L2 正则化项后，损失函数变为：

$$
J(\mathbf{\theta}) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(h_\theta(\mathbf{x}^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(\mathbf{x}^{(i)})) \right] + \frac{\lambda}{2m} \sum_{j=1}^{n} \theta_j^2
$$

其中：

- $\lambda$ 是正则化参数，控制正则化的强度。
- 注意：通常不对偏置项 $\theta_0$ 进行正则化。

**梯度更新公式：**

$$
\theta_j := \theta_j - \alpha \left( \frac{1}{m} \sum_{i=1}^{m} \left( h_\theta(\mathbf{x}^{(i)}) - y^{(i)} \right) x_j^{(i)} + \frac{\lambda}{m} \theta_j \right), \quad j \geq 1
$$

对于偏置项 $\theta_0$，更新规则为：

$$
\theta_0 := \theta_0 - \alpha \left( \frac{1}{m} \sum_{i=1}^{m} \left( h_\theta(\mathbf{x}^{(i)}) - y^{(i)} \right) x_0^{(i)} \right)
$$

#### 5.2 L1 正则化（套索回归）

L1 正则化使用参数的绝对值和作为正则化项：

$$
J(\mathbf{\theta}) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(h_\theta(\mathbf{x}^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(\mathbf{x}^{(i)})) \right] + \frac{\lambda}{m} \sum_{j=1}^{n} |\theta_j|
$$

**梯度更新公式**（注意 L1 正则化引入了非光滑性，常使用次梯度方法）：

$$
\theta_j := \theta_j - \alpha \left( \frac{1}{m} \sum_{i=1}^{m} \left( h_\theta(\mathbf{x}^{(i)}) - y^{(i)} \right) x_j^{(i)} + \lambda \cdot \text{sign}(\theta_j) \right), \quad j \geq 1
$$

### 6. 特征缩放

特征缩放（如标准化或归一化）可以加快梯度下降法的收敛速度，尤其是在特征具有不同尺度时。常用的方法包括：

- **标准化（Standardization）**：

  $$
  x_j := \frac{x_j - \mu_j}{\sigma_j}
  $$

  其中，$\mu_j$ 和 $\sigma_j$ 分别是第 $j$ 个特征的均值和标准差。

- **归一化（Normalization）**：

  $$
  x_j := \frac{x_j - x_{\text{min},j}}{x_{\text{max},j} - x_{\text{min},j}}
  $$

### 7. 多分类扩展

虽然逻辑回归本质上是用于二分类问题，但通过一些扩展方法，如**一对多（One-vs-Rest）**或**Softmax回归（多项逻辑回归）**，可以处理多分类问题。

#### 7.1 Softmax回归（多项逻辑回归）

对于 $K$ 类分类问题，Softmax函数将输入转换为每个类别的概率：

$$
h_\theta^{(k)}(\mathbf{x}) = \frac{e^{\mathbf{\theta}^{(k)T} \mathbf{x}}}{\sum_{c=1}^{K} e^{\mathbf{\theta}^{(c)T} \mathbf{x}}}, \quad k = 1, 2, \dots, K
$$

损失函数（交叉熵）为：

$$
J(\mathbf{\Theta}) = -\frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{K} y^{(k)(i)} \log(h_\theta^{(k)}(\mathbf{x}^{(i)}))
$$

其中：

- $\mathbf{\Theta} = [\mathbf{\theta}^{(1)}, \mathbf{\theta}^{(2)}, \dots, \mathbf{\theta}^{(K)}]$ 是模型参数矩阵。
- $y^{(k)(i)}$ 是指示变量，如果样本 $i$ 属于类别 $k$，则 $y^{(k)(i)} = 1$，否则为0。
