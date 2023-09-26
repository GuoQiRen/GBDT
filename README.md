## Gradient Boosting Decision Tree梯度决策提升树

### 一、GBDT树简介

**GBDT(Gradient Boosting Decision Tree)** 又叫 **MART（Multiple Additive Regression Tree)**，是一种迭代的决策树算法，该算法由多棵决策树组成，所有树的结论累加起来做最终答案。

单个决策树容易过拟合，但我们可以通过各种方法，抑制决策树的复杂性，降低单颗决策树的拟合能力，然后通过其他手段来集成多个决策树，最终能够很好的解决过拟合的问题。决策树可以认为是 if-then 规则的集合，易于理解，可解释性强，预测速度快。同时，决策树算法相比于其他的算法需要更少的特征工程，比如可以不用做特征标准化，可以很好的处理字段缺失的数据，也可以不用关心特征间是否相互依赖等。GBDT的核心在于**累加所有树的结果作为最终结果**，而不是分类树，分类树的结果累加显然是不正确的。

上面说的手段就是Boosting。Boosting是一族可将弱学习器提升为强学习器的算法，属于集成学习（ensemble learning）的范畴。**基于梯度提升算法的学习器**叫做 GBM(Gradient Boosting Machine)。弱决策树们通过梯度提升（Gradient Boosting）的方法，提升模型准确度。由此可见，梯度提升方法和决策树学习算法是一对完美的搭档。

### 二、加法模型

GBDT 算法可以看成是由 K 棵树组成的加法模型。加法模型的通常表达：
$$
f(x) = \sum_{m=1}^M \beta_mb(x;r_m)
$$
其中，$\beta_m$为基数函数的系数。$b(x;\gamma_m)$代表基函数，$\gamma_m$代表基函数参数。

在给定训练数据以及损失函数$L(y, f(x))$的条件下，学习加法模型$f(x)$成为**经验风险极小化即损失函数极小化问题**:
$$
min_{\beta_m, \gamma_m} \sum_{i=1}^NL(y_i, \sum_{m=1}^M \beta_mb(x;r_m))
$$

### 三、前向分步算法

解决加法模型的优化问题，可以用前向分布算法（forward stagewise algorithm）因为学习的是加法模型，如果能够从前往后，每一步只学习一个基函数及其系数（结构），逐步逼近优化目标函数，那么就可以简化复杂度。具体地， 每步只需要优化上面的损失函数，具体流程如下：

1.初始化$f_0(x) = 0$

2.对m=1,2,3,..., M

​	(a).极小化损失函数:$(\beta_m, \gamma_m) = argmin_{\beta, \gamma}\sum_{i=1}^NL(y_i, \sum_{m=1}^M \beta_mb(x;r_m))$

​	得到参数$\beta_m, \gamma_m$

​	(b).更新

​	$f_m(x) = f_{m-1}(x) + \beta_mb(x;r_m)$

3.得到加法模型

$f(x) = f_M(x) = \sum_{m=1}^M\beta_mb(x;r_m)$

这样，前向分步算法将同时求解从m=1到M所有参数$\beta_m, \gamma_m$的优化问题简化为逐次求解各个$\beta_m, \gamma_m$的优化问题。

### 四、梯度提升树

提升树算法采用前向分步算法。首先确定初始提升树$f_0(x)$ = 0，第m步的模型是：
$$
f_m(x) = f_{m-1}(x) + T(x;\gamma)
$$
其中，$f_{m-1}(x)$为当前模型，通过经验风险极小化确定下一个决策树的参数$\gamma$
$$
\gamma = argmin \sum_{i=1}^N L(y_i, f_{m-1}(x) + T(x;\gamma))
$$
针对不同问题的提升树学习算法，损失函数的选择也不同。

#### 4.1、负梯度拟合

在梯度提升算法中负梯度也被称为伪残差（pseudo-residuals）。

提升树用加法模型与前向分布算法实现学习的优化过程。当损失函数为平方损失和指数损失函数时，优化过程是相对简单的，例如平方损失函数可通过最小二乘法优化。但如果是一般性损失函数，优化的问题比较难，Freidman提出了梯度提升算法。这是利用最速下降法的近似方法，**其关键是利用损失函数的负梯度在当前模型的值** ：
$$
-[\frac{\partial L(y, f(x_i))}{\partial f(x_i)}]_{f(x)=f_{m-1}(x)}
$$
作为回归问题在当前模型的残差的近似值，拟合一个回归树。为什么要拟合负梯度呢？这就涉及到泰勒公式和梯度下降法了。

#### 4.2、泰勒公式

定义：泰勒公式是在一个用函数在某点的信息描述其附近取值的公式。

公式：
$$
f(x) = \sum_{n=0}^\infty \frac{f^n(x_0)}{n!}(x-x_0)^n
$$
一阶泰勒展开式: $f(x) = f(x_0) + f^{’}(x_0)(x-x_0) + Error$

#### 4.3、梯度下降法

在机器学习任务中，最小化损失函数$L(\theta)$，其中$\theta$是要求解的模型参数，梯度下降法常用来求解这种无约束最优化问题，即迭代方法：选择初始值$\theta_0$，不断迭代更新$\theta$，进行损失函数极小化、迭代公式：
$$
\theta^t = \theta^{t-1} +  \triangle\theta
$$
将$L(\theta^t)$在$\theta_{t-1}$处进行一阶泰勒展开：
$$
L(\theta^t) = L(\theta^{t-1} + \triangle\theta) \approx L(\theta^{t-1}) + L^{’}(\theta^{t-1})\triangle\theta
$$
要使得$L(\theta_t) < L(\theta_{t-1})$，可取: $\triangle\theta = -\alpha L^{’}(\theta^{t-1})$，则$\theta^t = \theta^{t-1} - \alpha L^{’}(\theta^{t-1})$，这里$\alpha$代表步长，一般直接赋值为一个小数，例如0.01.

#### 4.4、GBDT算法总结

总结一下 GBDT 的学习算法：

![](GBDT.png)

算法步骤解释：

1. 初始化，估计使损失函数极小化的常数值，它是只有一个根节点的树，即$\gamma$是一个常数值。
2. 对每个弱分类器
    （a）计算损失函数的负梯度在当前模型的值，将它作为残差的估计；
    （b）估计回归树叶节点区域，以拟合残差的近似值；
    （c）利用线性搜索估计叶节点区域的值，使损失函数极小化；
    （d）更新回归树。
3. 得到输出的最终模型$f(x)$