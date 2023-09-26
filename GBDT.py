import numpy as np


def create_dataset():
    """
    创建数据
    """
    x_train = np.array([[1, 5.56],
                        [2, 5.70],
                        [3, 5.91],
                        [4, 6.40],
                        [5, 6.80],
                        [6, 7.05],
                        [7, 8.90],
                        [8, 8.70],
                        [9, 9.00],
                        [10, 9.05]])
    x_test = np.array([[2], [5]])
    return x_train, x_test


def calculate_MSE(data_set):
    """
    计算CART回归树的节点方差Squared Error
    :param data_set: 数据集，包含目标列。  np.array，m*(n+1)
    :return: 当前节点（目标列）的方差
    """
    if data_set.shape[0] == 0:  # 如果输入一个空数据集，则返回0
        return 0
    return np.var(data_set[:, -1]) * data_set.shape[0]  # 方差=均方差*样本数量


def split_dataset(data_set, feature, value):
    """
    根据给定特征值，二分数据集。
    :param data_set: 同上
    :param feature: 待划分特征。因为是处理回归问题，这里我们假定数据集的特征都是连续型
    :param value: 阀值
    :return: 特征值小于等于阀值或大于阀值的两个子数据集. k*(n+1), (m-k)*(n+1)
    """
    arr1 = data_set[np.nonzero(data_set[:, feature] <= value)[0], :]  # 利用np.nonzero返回目标样本的索引值
    arr2 = data_set[np.nonzero(data_set[:, feature] > value)[0], :]
    return arr1, arr2


def choose_best_feature(data_set):
    """
    通过比较所有节点的方差和，选出方差和最小的特征与对应的特征值
    :param data_set: 同上
    :return: 最佳划分点和划分值
    """
    n = data_set.shape[1] - 1  # m是样本数量，n是特征数量

    min_loss = np.inf  # 初始化最小方差为无穷大的正数
    best_feature, best_value = 0, 0  # 声明变量的类型: 最佳分割特征、最佳分割值

    for feature in range(n):

        values = set(data_set[:, feature].tolist())  # 选取所有出现过的值作为阀值

        for value in values:

            left_set, right_set = split_dataset(data_set, feature, value)

            left_loss = calculate_MSE(left_set)
            right_loss = calculate_MSE(right_set)

            new_loss = left_loss + right_loss

            # 选取方差和最小的特征和对应的阀值
            if new_loss < min_loss:
                min_loss = new_loss
                best_feature = feature
                best_value = value

    return best_feature, best_value


def calculate_leaf(data_set):
    """
    计算当前节点的目标列均值（作为当前节点的预测值）
    预测值的计算具体是要根据损失函数确定的。
    不用的损失函数，对应不同的叶子节点值。
    平方误差损失的节点值是均值。
    :param data_set: 同上
    :return: 目标列均值
    """
    return np.mean(data_set[:, -1])


class GradientBoostingDecisionTree:
    """
    cart回归树作弱学习器，平方误差函数作损失函数。Loss = 1/2*(y-h(x))**2
    """

    def create_tree(self, data_set, max_depth=4):
        """
        创建CART回归树
        :param data_set: 同上
        :param max_depth: 设定回归树的最大深度，防止无限生长（过拟合）
        :return: 字典形式的cart回归树模型
        """
        if len(set(data_set[:, -1].tolist())) == 1:  # 如果当前节点的值都相同，结束递归
            return calculate_leaf(data_set)

        if max_depth == 1:  # 如果层数超出设定层数，结束递归
            return calculate_leaf(data_set)

        # 创建回归树
        best_feature, best_value = choose_best_feature(data_set)
        my_tree = {'FeatureIndex': best_feature, 'FeatureValue': best_value}

        # 正式分割数据集
        left_set, right_set = split_dataset(data_set, best_feature, best_value)

        # 决策树的左集合与右集合
        my_tree['left'] = self.create_tree(left_set, max_depth - 1)  # 存储左子树的信息
        my_tree['right'] = self.create_tree(right_set, max_depth - 1)  # 存储右子树的信息

        return my_tree

    def predict_by_cart(self, cart_tree, test_data):
        """
        根据训练好的cart回归树，预测待测数据的值
        """
        if not isinstance(cart_tree, dict):  # 不是字典，意味着到了叶子结点，此时返回叶子结点的值即可
            return cart_tree

        feature_index = cart_tree['FeatureIndex']  # 获取回归树的第一层特征索引
        feature_value = test_data[feature_index]  # 根据特征索引找到待测数据对应的特征值， 作为下面是进入左子树还是右子树的依据

        if feature_value <= cart_tree['FeatureValue']:
            return self.predict_by_cart(cart_tree['left'], test_data)
        elif feature_value > cart_tree['FeatureValue']:
            return self.predict_by_cart(cart_tree['right'], test_data)

    def predict_all_by_cart(self, cart_tree, test_data):
        """
        根据训练好的cart回归树预测所有待测数据的值
        """
        test_data = np.array(test_data)
        predictions = np.zeros(test_data.shape[0])
        for i in range(test_data.shape[0]):
            predictions[i] = self.predict_by_cart(cart_tree, test_data[i])
        return predictions

    def GBDT_main(self, data_set, tree_num=4):
        """
        训练GBDT回归树,简化版本，这里没根据Shrinkage思想添加正则化参数step
        """
        fx_pre = np.mean(data_set[:, -1])  # 记录前m-1个模型的预测值之均值
        weak_trees = [fx_pre]  # 弱学习器的列表
        targets = data_set[:, -1].copy()  # 存储训练集的目标值

        for i in range(tree_num):
            data_set[:, -1] = targets - fx_pre  # 平方误差损失的一阶导的负数（负梯度）正好是目标值与预测值的差
            my_tree = self.create_tree(data_set)  # 把原目标列替换为残差后，训练弱学习器

            weak_trees.append(my_tree)

            fx_pre += self.predict_all_by_cart(my_tree, data_set[:, :-1])  # 计算所有模型的预测值之和（针对训练集）
            loss = np.var(targets - fx_pre) * targets.shape[0]  # 计算损失函数

            print('Iter:%d, Loss: %.6f' % (i + 1, loss))

            if loss == 0:  # 损失为0，跳出循环。或者可以设定一个阀值，当小于该阀值的时候，退出循环
                break

        return weak_trees

    def predict(self, tree, test_data):
        """
        根据训练好的GDBT模型，预测待测数据
        """
        predict = tree[0]  # 模型的初始化值f0x
        for cart in tree[1:]:  # cart树从索引1开始
            predict += self.predict_all_by_cart(cart, test_data)  # 累加预测结果
        return predict


if __name__ == '__main__':
    # 创建数据集
    train_x, test_x = create_dataset()

    # 训练
    gTree = GradientBoostingDecisionTree()
    model = gTree.GBDT_main(train_x)

    # 预测
    pred_y = gTree.predict(model, test_x)
    print(pred_y)
