#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Awesome-DL-Models 
@File    :decision_tree.py
@Author  :JackHCC
@Date    :2022/1/18 18:54 
@Desc    :Decision Tree C4.5 and CART

'''
import json
import collections
import numpy as np
from math import log
from collections import Counter


def entropy(y, base=2):
    """计算随机变量Y的熵"""
    count = collections.Counter(y)
    ans = 0
    for freq in count.values():
        prob = freq / len(y)
        ans -= prob * log(prob, base)
    return ans


def conditional_entropy(x, y, base=2):
    """计算随机变量X给定的条件下随机变量Y的条件熵H(Y|X)"""
    freq_y_total = collections.defaultdict(collections.Counter)  # 统计随机变量X取得每一个取值时随机变量Y的频数
    freq_x = collections.Counter()  # 统计随机变量X每一个取值的频数
    for i in range(len(x)):
        freq_y_total[x[i]][y[i]] += 1
        freq_x[x[i]] += 1
    ans = 0
    for xi, freq_y_xi in freq_y_total.items():
        res = 0
        for freq in freq_y_xi.values():
            prob = freq / freq_x[xi]
            res -= prob * log(prob, base)
        ans += res * (freq_x[xi] / len(x))
    return ans


def information_gain(x, y, idx, base=2):
    """计算特征A(第idx个特征)对训练数据集D(输入数据x,输出数据y)的信息增益"""
    return entropy(y, base=base) - conditional_entropy([x[i][idx] for i in range(len(x))], y, base=base)


def information_gain_ratio(x, y, idx, base=2):
    """计算特征A(第idx个特征)对训练数据集D(输入数据x,输出数据y)的信息增益比"""
    return information_gain(x, y, idx, base=base) / entropy([x[i][idx] for i in range(len(x))], base=base)


# C4.5 Node
class Node:
    def __init__(self, node_type, class_name, feature_name=None,
                 info_gain_ratio_value=0.0):
        # 结点类型（internal或leaf）
        self.node_type = node_type
        # 特征名
        self.feature_name = feature_name
        # 类别名
        self.class_name = class_name
        # 子结点树
        self.child_nodes = []
        # 熵增益
        self.info_gain_ratio_value = info_gain_ratio_value

    def __repr__(self):
        return json.dumps(self, indent=3, default=lambda obj: obj.__dict__, ensure_ascii=False)

    def add_sub_tree(self, key, sub_tree):
        self.child_nodes.append({"condition": key, "sub_tree": sub_tree})


class DecisionTreeID3WithoutPruning:
    """ID3生成算法构造的决策树（仅支持离散型特征）-不包括剪枝"""

    class Node:
        def __init__(self, mark, use_feature=None, children=None):
            if children is None:
                children = {}
            self.mark = mark
            self.use_feature = use_feature  # 用于分类的特征
            self.children = children  # 子结点

        @property
        def is_leaf(self):
            return len(self.children) == 0

    def __init__(self, x, y, labels=None, base=2, epsilon=0):
        if labels is None:
            labels = ["特征{}".format(i + 1) for i in range(len(x[0]))]
        self.labels = labels  # 特征的标签
        self.base = base  # 熵的单位（底数）
        self.epsilon = epsilon  # 决策树生成的阈值

        # ---------- 构造决策树 ----------
        self.n = len(x[0])
        self.root = self._build(x, y, set(range(self.n)))  # 决策树生成

    def _build(self, x, y, spare_features_idx):
        """根据当前数据构造结点

        :param x: 输入变量
        :param y: 输出变量
        :param spare_features_idx: 当前还可以使用的特征的下标
        """
        freq_y = collections.Counter(y)

        # 若D中所有实例属于同一类Ck，则T为单结点树，并将Ck作为该结点的类标记
        if len(freq_y) == 1:
            return self.Node(y[0])

        # 若A为空集，则T为单结点树，并将D中实例数最大的类Ck作为该结点的标记
        if not spare_features_idx:
            return self.Node(freq_y.most_common(1)[0][0])

        # 计算A中各特征对D的信息增益，选择信息增益最大的特征Ag
        best_feature_idx, best_gain = -1, 0
        for feature_idx in spare_features_idx:
            gain = self.information_gain(x, y, feature_idx)
            if gain > best_gain:
                best_feature_idx, best_gain = feature_idx, gain

        # 如果Ag的信息增益小于阈值epsilon，则置T为单结点树，并将D中实例数最大的类Ck作为该结点的类标记
        if best_gain <= self.epsilon:
            return self.Node(freq_y.most_common(1)[0][0])

        # 依Ag=ai将D分割为若干非空子集Di，将Di中实例数最大的类作为标记，构建子结点
        node = self.Node(freq_y.most_common(1)[0][0], use_feature=best_feature_idx)
        features = set()
        sub_x = collections.defaultdict(list)
        sub_y = collections.defaultdict(list)
        for i in range(len(x)):
            feature = x[i][best_feature_idx]
            features.add(feature)
            sub_x[feature].append(x[i])
            sub_y[feature].append(y[i])

        for feature in features:
            node.children[feature] = self._build(sub_x[feature], sub_y[feature],
                                                 spare_features_idx - {best_feature_idx})
        return node

    def __repr__(self):
        """深度优先搜索绘制可视化的决策树"""

        def dfs(node, depth=0, value=""):
            if node.is_leaf:  # 处理叶结点的情况
                res.append(value + " -> " + node.mark)
            else:
                if depth > 0:  # 处理中间结点的情况
                    res.append(value + " :")
                for val, child in node.children.items():
                    dfs(child, depth + 1, "  " * depth + self.labels[node.use_feature] + " = " + val)

        res = []
        dfs(self.root)
        return "\n".join(res)

    def information_gain(self, x, y, idx):
        """计算信息增益"""
        return entropy(y, base=self.base) - conditional_entropy([x[i][idx] for i in range(len(x))], y, base=self.base)


class DecisionTreeID3:
    """ID3生成算法构造的决策树（仅支持离散型特征）"""

    class Node:
        def __init__(self, mark, ee, use_feature=None, children=None):
            if children is None:
                children = {}
            self.mark = mark
            self.use_feature = use_feature  # 用于分类的特征
            self.children = children  # 子结点
            self.ee = ee  # 以当前结点为叶结点的经验熵

        @property
        def is_leaf(self):
            return len(self.children) == 0

    def __init__(self, x, y, labels=None, base=2, epsilon=0, alpha=0.05):
        if labels is None:
            labels = ["特征{}".format(i + 1) for i in range(len(x[0]))]
        self.labels = labels  # 特征的标签
        self.base = base  # 熵的单位（底数）
        self.epsilon = epsilon  # 决策树生成的阈值
        self.alpha = alpha  # 决策树剪枝的参数

        # ---------- 构造决策树 ----------
        self.n = len(x[0])
        self.root = self._build(x, y, set(range(self.n)))  # 决策树生成
        self._pruning(self.root)  # 决策树剪枝

    def _build(self, x, y, spare_features_idx):
        """根据当前数据构造结点

        :param x: 输入变量
        :param y: 输出变量
        :param spare_features_idx: 当前还可以使用的特征的下标
        """
        freq_y = collections.Counter(y)
        ee = entropy(y, base=self.base)  # 计算以当前结点为叶结点的经验熵

        # 若D中所有实例属于同一类Ck，则T为单结点树，并将Ck作为该结点的类标记
        if len(freq_y) == 1:
            return self.Node(y[0], ee)

        # 若A为空集，则T为单结点树，并将D中实例数最大的类Ck作为该结点的标记
        if not spare_features_idx:
            return self.Node(freq_y.most_common(1)[0][0], ee)

        # 计算A中各特征对D的信息增益，选择信息增益最大的特征Ag
        best_feature_idx, best_gain = -1, 0
        for feature_idx in spare_features_idx:
            gain = self.information_gain(x, y, feature_idx)
            if gain > best_gain:
                best_feature_idx, best_gain = feature_idx, gain

        # 如果Ag的信息增益小于阈值epsilon，则置T为单结点树，并将D中实例数最大的类Ck作为该结点的类标记
        if best_gain <= self.epsilon:
            return self.Node(freq_y.most_common(1)[0][0], ee)

        # 依Ag=ai将D分割为若干非空子集Di，将Di中实例数最大的类作为标记，构建子结点
        node = self.Node(freq_y.most_common(1)[0][0], ee, use_feature=best_feature_idx)
        features = set()
        sub_x = collections.defaultdict(list)
        sub_y = collections.defaultdict(list)
        for i in range(len(x)):
            feature = x[i][best_feature_idx]
            features.add(feature)
            sub_x[feature].append(x[i])
            sub_y[feature].append(y[i])

        for feature in features:
            node.children[feature] = self._build(sub_x[feature], sub_y[feature],
                                                 spare_features_idx - {best_feature_idx})
        return node

    def _pruning(self, node):
        # 处理当前结点为叶结点的情况：不剪枝，直接返回
        if node.is_leaf:
            return 1, node.ee

        # 计算剪枝（以当前结点为叶结点）的损失函数
        loss1 = node.ee + 1 * self.alpha

        # 计算不剪枝的损失函数
        num, ee = 1, 0
        for child in node.children.values():
            child_num, child_ee = self._pruning(child)
            num += child_num
            ee += child_ee
        loss2 = ee + num * self.alpha

        # 处理需要剪枝的情况
        if loss1 < loss2:
            node.children = {}
            return 1, node.ee

        # 处理不需要剪枝的情况
        else:
            return num, ee

    def __repr__(self):
        """深度优先搜索绘制可视化的决策树"""

        def dfs(node, depth=0, value=""):
            if node.is_leaf:  # 处理叶结点的情况
                res.append(value + " -> " + node.mark)
            else:
                if depth > 0:  # 处理中间结点的情况
                    res.append(value + " :")
                for val, child in node.children.items():
                    dfs(child, depth + 1, "  " * depth + self.labels[node.use_feature] + " = " + val)

        res = []
        dfs(self.root)
        return "\n".join(res)

    def information_gain(self, x, y, idx):
        """计算信息增益"""
        return entropy(y, base=self.base) - conditional_entropy([x[i][idx] for i in range(len(x))], y, base=self.base)


# C4.5
class DecisionTreeC45:
    def __init__(self, epsilon):
        self.epsilon = epsilon
        self.tree = None

    def fit(self, train_set, y, feature_names):
        features_indices = list(range(len(feature_names)))
        self.tree = self._fit(train_set, y, features_indices, feature_names)
        return self

    # C4.5
    def _fit(self, train_data, y, features_indices, feature_labels):
        LEAF = 'leaf'
        INTERNAL = 'internal'
        class_num = len(np.unique(y))

        # （1）如果训练数据集所有实例都属于同一类Ck
        label_set = set(y)
        if len(label_set) == 1:
            # 将Ck作为该结点的类
            return Node(node_type=LEAF, class_name=label_set.pop())

        # （2）如果特征集为空
        # 计算每一个类出现的个数
        class_len = Counter(y).most_common()
        (max_class, max_len) = class_len[0]

        if len(features_indices) == 0:
            # 将实例数最大的类Ck作为该结点的类
            return Node(LEAF, class_name=max_class)

        # （3）计算信息增益，并选择信息增益最大的特征
        max_feature = 0
        max_gda = 0
        D = y.copy()
        # 计算特征集A中各特征
        for feature in features_indices:
            # 选择训练集中的第feature列（即第feature个特征）
            A = np.array(train_data[:, feature].flat)
            # 计算信息增益
            gda = self._calc_ent_grap(A, D)
            if self._calc_ent(A) != 0:
                # 计算信息增益比
                gda /= self._calc_ent(A)
            # 选择信息增益最大的特征Ag
            if gda > max_gda:
                max_gda, max_feature = gda, feature

        # （4）如果Ag信息增益小于阈值
        if max_gda < self.epsilon:
            # 将训练集中实例数最大的类Ck作为该结点的类
            return Node(LEAF, class_name=max_class)

        max_feature_label = feature_labels[max_feature]

        # （6）移除已选特征Ag
        sub_feature_indecs = np.setdiff1d(features_indices, max_feature)
        sub_feature_labels = np.setdiff1d(feature_labels, max_feature_label)

        # （5）构建非空子集
        # 构建结点
        feature_name = feature_labels[max_feature]
        tree = Node(INTERNAL, class_name=None, feature_name=feature_name,
                    info_gain_ratio_value=max_gda)

        max_feature_col = np.array(train_data[:, max_feature].flat)
        # 将类按照对应的实例数递减顺序排列
        feature_value_list = [x[0] for x in Counter(max_feature_col).most_common()]
        # 遍历Ag的每一个可能值ai
        for feature_value in feature_value_list:
            index = []
            for i in range(len(y)):
                if train_data[i][max_feature] == feature_value:
                    index.append(i)

            # 递归调用步（1）~步（5），得到子树
            sub_train_set = train_data[index]
            sub_train_label = y[index]
            sub_tree = self._fit(sub_train_set, sub_train_label, sub_feature_indecs, sub_feature_labels)
            # 在结点中，添加其子结点构成的树
            tree.add_sub_tree(feature_value, sub_tree)
        return tree

    # 计算数据集x的经验熵H(x)
    @staticmethod
    def _calc_ent(x):
        x_value_list = set([x[i] for i in range(x.shape[0])])
        ent = 0.0
        for x_value in x_value_list:
            p = float(x[x == x_value].shape[0]) / x.shape[0]
            logp = np.log2(p)
            ent -= p * logp
        return ent

    # 计算条件熵H(y/x)
    def _calc_condition_ent(self, x, y):
        x_value_list = set([x[i] for i in range(x.shape[0])])
        ent = 0.0
        for x_value in x_value_list:
            sub_y = y[x == x_value]
            temp_ent = self._calc_ent(sub_y)
            ent += (float(sub_y.shape[0]) / y.shape[0]) * temp_ent
        return ent

    # 计算信息增益
    def _calc_ent_grap(self, x, y):
        base_ent = self._calc_ent(y)
        condition_ent = self._calc_condition_ent(x, y)
        ent_grap = base_ent - condition_ent
        return ent_grap

    def __repr__(self):
        return str(self.tree)


class CartNode:
    def __init__(self, value, feature, left=None, right=None):
        self.value = value.tolist()
        self.feature = feature.tolist()
        self.left = left
        self.right = right

    def __repr__(self):
        return json.dumps(self, indent=3, default=lambda obj: obj.__dict__, ensure_ascii=False)


class LeastSquareRegTree:
    def __init__(self, train_X, y, epsilon):
        # 训练集特征值
        self.x = train_X
        # 类别
        self.y = y
        # 特征总数
        self.feature_count = train_X.shape[1]
        # 损失阈值
        self.epsilon = epsilon
        # 回归树
        self.tree = None

    def _fit(self, x, y, feature_count):
        # （1）选择最优切分点变量j与切分点s，得到选定的对(j,s)，并解得c1，c2
        (j, s, minval, c1, c2) = self._divide(x, y, feature_count)
        # 初始化树
        tree = CartNode(feature=j, value=x[s, j], left=None, right=None)
        # 用选定的对(j,s)划分区域，并确定响应的输出值
        if minval < self.epsilon or len(y[np.where(x[:, j] <= x[s, j])]) <= 1:
            tree.left = c1
        else:
            # 对左子区域调用步骤（1）、（2）
            tree.left = self._fit(x[np.where(x[:, j] <= x[s, j])],
                                  y[np.where(x[:, j] <= x[s, j])],
                                  self.feature_count)
        if minval < self.epsilon or len(y[np.where(x[:, j] > s)]) <= 1:
            tree.right = c2
        else:
            # 对右子区域调用步骤（1）、（2）
            tree.right = self._fit(x[np.where(x[:, j] > x[s, j])],
                                   y[np.where(x[:, j] > x[s, j])],
                                   self.feature_count)
        return tree

    def fit(self):
        self.tree = self._fit(self.x, self.y, self.feature_count)
        return self

    @staticmethod
    def _divide(x, y, feature_count):
        # 初始化损失误差
        cost = np.zeros((feature_count, len(x)))
        # 公式5.21
        for i in range(feature_count):
            for k in range(len(x)):
                # k行i列的特征值
                value = x[k, i]
                y1 = y[np.where(x[:, i] <= value)]
                c1 = np.mean(y1)
                y2 = y[np.where(x[:, i] > value)]
                if len(y2) == 0:
                    c2 = 0
                else:
                    c2 = np.mean(y2)
                y1[:] = y1[:] - c1
                y2[:] = y2[:] - c2
                cost[i, k] = np.sum(y1 * y1) + np.sum(y2 * y2)
        # 选取最优损失误差点
        cost_index = np.where(cost == np.min(cost))
        # 所选取的特征
        j = cost_index[0][0]
        # 选取特征的切分点
        s = cost_index[1][0]
        # 求两个区域的均值c1,c2
        c1 = np.mean(y[np.where(x[:, j] <= x[s, j])])
        c2 = np.mean(y[np.where(x[:, j] > x[s, j])])
        return j, s, cost[cost_index], c1, c2

    def __repr__(self):
        return str(self.tree)


if __name__ == '__main__':
    feature_names = np.array(["年龄", "有工作", "有自己的房子", "信贷情况"])
    X_train = np.array([
        ["青年", "否", "否", "一般"],
        ["青年", "否", "否", "好"],
        ["青年", "是", "否", "好"],
        ["青年", "是", "是", "一般"],
        ["青年", "否", "否", "一般"],
        ["中年", "否", "否", "一般"],
        ["中年", "否", "否", "好"],
        ["中年", "是", "是", "好"],
        ["中年", "否", "是", "非常好"],
        ["中年", "否", "是", "非常好"],
        ["老年", "否", "是", "非常好"],
        ["老年", "否", "是", "好"],
        ["老年", "是", "否", "好"],
        ["老年", "是", "否", "非常好"],
        ["老年", "否", "否", "一般"]
    ])
    y = np.array(["否", "否", "是", "是", "否",
                  "否", "否", "是", "是", "是",
                  "是", "是", "是", "是", "否"])

    print("开始测试熵，条件熵，信息增益，信息增益比……")
    print("熵：", entropy(y))
    print("条件熵：", conditional_entropy([X_train[i][0] for i in range(len(X_train))], y))
    print("信息增益：", information_gain(X_train, y, idx=0))
    print("信息增益比 gR(D,A1)：", information_gain_ratio(X_train, y, idx=0))
    print("信息增益比 gR(D,A2)：", information_gain_ratio(X_train, y, idx=1))
    print("信息增益比 gR(D,A3)：", information_gain_ratio(X_train, y, idx=2))
    print("信息增益比 gR(D,A4)：", information_gain_ratio(X_train, y, idx=3))

    print("------------------------------------------")
    print("开始测试ID3算法 - 不含剪枝……")
    decision_tree = DecisionTreeID3WithoutPruning(X_train, y, labels=["年龄", "有工作", "有自己的房子", "信贷情况"])
    print(decision_tree)

    print("------------------------------------------")
    print("开始测试ID3算法生成决策树 - 包含剪枝……")
    decision_tree = DecisionTreeID3(X_train, y, labels=["年龄", "有工作", "有自己的房子", "信贷情况"], alpha=0.2)
    print(decision_tree)
    decision_tree = DecisionTreeID3(X_train, y, labels=["年龄", "有工作", "有自己的房子", "信贷情况"], alpha=0.3)
    print(decision_tree)

    print("------------------------------------------")
    print("开始测试C4.5算法……")
    dt_tree = DecisionTreeC45(epsilon=0.1)
    dt_tree.fit(X_train, y, feature_names)
    print(dt_tree)

    print("------------------------------------------")
    print("开始测试CART最小二乘回归算法……")
    # Test CART
    train_X = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]).T
    y = np.array([4.50, 4.75, 4.91, 5.34, 5.80, 7.05, 7.90, 8.23, 8.70, 9.00])

    model_tree = LeastSquareRegTree(train_X, y, epsilon=0.2)
    model_tree.fit()
    print(model_tree)
