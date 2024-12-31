

## 线性回归的数学推导

### 1. 引言

线性回归是一种基本且广泛应用的监督学习算法，主要用于预测连续型目标变量。它假设输入特征与目标变量之间存在线性关系。

### 2. 假设函数

在 $m$ 个特征的情况下，线性回归的假设函数（预测函数）定义为：

$$
h_\theta(\mathbf{x}) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \dots + \theta_m x_m = \mathbf{\theta}^T \mathbf{x}
$$

其中：

- $h_\theta(\mathbf{x})$ 是预测值。
- $\mathbf{\theta} = [\theta_0, \theta_1, \dots, \theta_m]^T$ 是模型参数。
- $\mathbf{x} = [1, x_1, x_2, \dots, x_m]^T$ 是输入特征向量（包括偏置项 $x_0 = 1$）。

### 3. 损失函数

为了衡量预测值与真实值之间的差距，我们使用**均方误差（Mean Squared Error, MSE）**作为损失函数。损失函数定义为：

$$
J(\mathbf{\theta}) = \frac{1}{2m} \sum_{i=1}^{m} \left( h_\theta(\mathbf{x}^{(i)}) - y^{(i)} \right)^2
$$

其中：

- $m$ 是样本数量。
- $h_\theta(\mathbf{x}^{(i)})$ 是第 $i$ 个样本的预测值。
- $y^{(i)}$ 是第 $i$ 个样本的真实值。

**为何选择均方误差？**

均方误差不仅简单且具有凸性，确保了全局最小值的存在，使得优化过程更加稳定和高效。

### 4. 参数优化

我们的目标是找到一组参数 $\mathbf{\theta}$，使得损失函数 $J(\mathbf{\theta})$ 最小。常用的优化方法有梯度下降法和正规方程法。这里我们详细推导梯度下降法。

#### 4.1 梯度下降法

梯度下降法是一种迭代优化算法，通过不断更新参数向最小值的方向前进。

**参数更新规则：**

$$
\theta_j := \theta_j - \alpha \frac{\partial J(\mathbf{\theta})}{\partial \theta_j}
$$

其中：

- $\alpha$ 是学习率，控制步长的大小。
- $\frac{\partial J(\mathbf{\theta})}{\partial \theta_j}$ 是损失函数对参数 $\theta_j$ 的偏导数。

**计算偏导数：**

首先，对损失函数 $J(\mathbf{\theta})$ 关于参数 $\theta_j$ 求偏导：

$$
\frac{\partial J(\mathbf{\theta})}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} \left( h_\theta(\mathbf{x}^{(i)}) - y^{(i)} \right) x_j^{(i)}
$$

因此，参数更新规则可以具体表示为：

$$
\theta_j := \theta_j - \alpha \left( \frac{1}{m} \sum_{i=1}^{m} \left( h_\theta(\mathbf{x}^{(i)}) - y^{(i)} \right) x_j^{(i)} \right)
$$

这一过程会对所有参数 $\theta_j$（包括偏置项 $\theta_0$）同时进行更新，直到损失函数收敛到最小值。

### 5. 正规方程法

相比于梯度下降法，正规方程法可以直接求解出一组参数使损失函数达到最小值，而无需进行迭代。正规方程法基于线性代数中的最小二乘法。

**正规方程的推导：**

为了使损失函数 $J(\mathbf{\theta})$ 最小化，我们设其偏导数为零：

$$
\frac{\partial J(\mathbf{\theta})}{\partial \mathbf{\theta}} = 0
$$

将损失函数代入，得到：

$$
\frac{1}{m} \mathbf{X}^T (\mathbf{X}\mathbf{\theta} - \mathbf{y}) = 0
$$

通过整理，可以得到正规方程：

$$
\mathbf{\theta} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}
$$

其中：

- $\mathbf{X}$ 是包含所有训练样本的特征矩阵，形状为 $(m, n+1)$。
- $\mathbf{y}$ 是目标变量向量，形状为 $(m, 1)$。
- $\mathbf{\theta}$ 是参数向量，形状为 $(n+1, 1)$。

**优势与劣势：**

- **优势：**
  - 能够直接求解出最优参数，无需选择学习率或确定迭代次数。
- **劣势：**
  - 计算复杂度高，特别是当特征数量较多时，计算 $(\mathbf{X}^T \mathbf{X})^{-1}$ 可能会非常耗时，甚至无法计算。

### 6. 特征缩放

在使用梯度下降法时，特征缩放（如标准化或归一化）可以加快收敛速度，避免某些特征对参数更新的影响过大。

**常用的方法：**

- **标准化（Standardization）：**
  
  将特征转换为均值为 $0$、标准差为 $1$ 的分布。
  
  $$
  x_j := \frac{x_j - \mu_j}{\sigma_j}
  $$
  
  其中，$\mu_j$ 和 $\sigma_j$ 分别是第 $j$ 个特征的均值和标准差。

- **归一化（Normalization）：**
  
  将特征缩放到一个固定范围（通常是 $[0, 1]$）。
  
  $$
  x_j := \frac{x_j - x_{j,\text{min}}}{x_{j,\text{max}} - x_{j,\text{min}}}
  $$

### 7. 正则化

为了防止模型过拟合，可以在损失函数中添加正则化项。常见的正则化方法有 L1 和 L2 正则化。

**L2 正则化（岭回归）：**

$$
J(\mathbf{\theta}) = \frac{1}{2m} \sum_{i=1}^{m} \left( h_\theta(\mathbf{x}^{(i)}) - y^{(i)} \right)^2 + \frac{\lambda}{2m} \sum_{j=1}^{n} \theta_j^2
$$

其中，$\lambda$ 是正则化参数，用于控制正则化的强度。

