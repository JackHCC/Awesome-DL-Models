#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Awesome-DL-Models 
@File    :knn.py
@Author  :JackHCC
@Date    :2022/1/15 19:26 
@Desc    : KD Tree construction and search

'''
import json
import logging


class Node:
    def __init__(self, value, index, left_child, right_child):
        self.value = value.tolist()
        self.index = index
        self.left_child = left_child
        self.right_child = right_child

    # 重写打印或对象信息
    def __repr__(self):
        return json.dumps(self, indent=3, default=lambda obj: obj.__dict__, ensure_ascii=False)


class KDTree:
    def __init__(self, data):
        self.data = np.asarray(data)
        self.kd_tree = None
        self._create_kd_tree(data)

    def _split_sub_tree(self, data, depth=0):
        # Stop until no instance of the subregion exists
        if len(data) == 0:
            return None

        # Select the tangent axis, starting from 0
        l = depth % data.shape[1]
        # Sort data
        data = data[data[:, l].argsort()]
        # Take the median of all instance coordinates as the tangent point
        median_index = data.shape[0] // 2
        # Gets the location of the node in the dataset
        node_index = [i for i, v in enumerate(self.data) if list(v) == list(data[median_index])]

        return Node(
            # This Mode
            value=data[median_index],
            # the location of the node in the dataset
            index=node_index[0],
            # left child node
            left_child=self._split_sub_tree(data[:median_index], depth + 1),
            # right child node
            right_child=self._split_sub_tree(data[median_index + 1:], depth + 1)
        )

    def _create_kd_tree(self, X):
        self.kd_tree = self._split_sub_tree(X)

    def query(self, data, k=1):
        data = np.asarray(data)
        hits = self._search(data, self.kd_tree, k=k, k_neighbor_sets=list())
        dd = np.array([hit[0] for hit in hits])
        ii = np.array([hit[1] for hit in hits])
        return dd, ii

    def __repr__(self):
        return str(self.kd_tree)

    @staticmethod
    def _cal_node_distance(node1, node2, method=2):
        if type(method) is not int:
            logging.error("method must be a integer type")
        if method == 2:
            dis = np.sqrt(np.sum(np.square(node1 - node2)))
        elif method == 1:
            dis = np.abs(np.sum(node1 - node2))
        elif method == 0:
            dis = np.max(node1 - node2)
        else:
            dis = np.power(np.sum(np.power(node1 - node2, method)), 1 / method)
        return dis

    def _search(self, point, tree=None, k=1, k_neighbor_sets=None, depth=0):
        if k_neighbor_sets is None:
            k_neighbor_sets = []
        if tree is None:
            return k_neighbor_sets

        # Find the leaf node containing the target point X
        if tree.left_child is None and tree.right_child is None:
            # Update the current k nearest neighbor point set
            return self._update_k_neighbor_sets(k_neighbor_sets, k, tree, point)

        # Recursively down the KD tree
        if point[0][depth % k] < tree.value[depth % k]:
            direct = 'left'
            next_branch = tree.left_child
        else:
            direct = 'right'
            next_branch = tree.right_child
        if next_branch is not None:
            # Judge the current node and update the current k nearest neighbor point set
            k_neighbor_sets = self._update_k_neighbor_sets(k_neighbor_sets, k, next_branch, point)
            # (3)(b)检查另一子结点对应的区域是否相交
            if direct == 'left':
                node_distance = self._cal_node_distance(point, tree.right_child.value)
                if k_neighbor_sets[0][0] > node_distance:
                    # Check whether the region corresponding to another child node intersects
                    return self._search(point, tree=tree.right_child, k=k, depth=depth + 1,
                                        k_neighbor_sets=k_neighbor_sets)
            else:
                node_distance = self._cal_node_distance(point, tree.left_child.value)
                if k_neighbor_sets[0][0] > node_distance:
                    return self._search(point, tree=tree.left_child, k=k, depth=depth + 1,
                                        k_neighbor_sets=k_neighbor_sets)

        return self._search(point, tree=next_branch, k=k, depth=depth + 1, k_neighbor_sets=k_neighbor_sets)

    def _update_k_neighbor_sets(self, best, k, tree, point):
        # Calculate the distance between the target point and the current node
        node_distance = self._cal_node_distance(point, tree.value)
        if len(best) == 0:
            best.append((node_distance, tree.index, tree.value))
        elif len(best) < k:
            # If the number of elements of "current k nearest neighbor point set" is less than k
            self._insert_k_neighbor_sets(best, tree, node_distance)
        else:
            # The leaf node distance is less than the farthest point distance in the current nearest neighbor point set
            if best[0][0] > node_distance:
                best = best[1:]
                self._insert_k_neighbor_sets(best, tree, node_distance)
        return best

    @staticmethod
    def _insert_k_neighbor_sets(best, tree, node_distance):
        """Put the farthest node in front"""
        n = len(best)
        for i, item in enumerate(best):
            if item[0] < node_distance:
                # Insert the farthest node to the front
                best.insert(i, (node_distance, tree.index, tree.value))
                break
        if len(best) == n:
            best.append((node_distance, tree.index, tree.value))


def print_k_neighbor_sets(k, ii, dd):
    if k == 1:
        text = "x点的最近邻点是"
    else:
        text = "x点的%d个近邻点是" % k

    for i, index in enumerate(ii):
        res = X_train[index]
        if i == 0:
            text += str(tuple(res))
        else:
            text += ", " + str(tuple(res))

    if k == 1:
        text += "，距离是"
    else:
        text += "，距离分别是"
    for i, dist in enumerate(dd):
        if i == 0:
            text += "%.4f" % dist
        else:
            text += ", %.4f" % dist

    print(text)


if __name__ == '__main__':
    import numpy as np
    print("开始测试KNN算法的KD树实现方案：构造与搜索……")
    X_train = np.array([[2, 3],
                        [5, 4],
                        [9, 6],
                        [4, 7],
                        [8, 1],
                        [7, 2]])
    kd_tree = KDTree(X_train)
    k = 3
    dists, indices = kd_tree.query(np.array([[3, 4.5]]), k=k)
    print_k_neighbor_sets(k, indices, dists)
    print(kd_tree)
