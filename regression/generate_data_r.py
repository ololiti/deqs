# -*- coding: utf-8 -*-
"""generate_data.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16NtQpiatQbGpMN6-wBANkPRh70b-a9Gm
"""

import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numexpr


# length has to be odd#
def generate(length, operators, numbers):
    if length == 1:
        return np.random.choice(numbers)

    if length == 3:
        return np.random.choice(numbers) + np.random.choice(operators) + np.random.choice(numbers)

    if (random.uniform(0, 1) >= 0.6):
        return '(' + generate(length - 2, operators, numbers) + ')'
    else:
        op = np.random.choice(operators)
        position = np.random.choice(range(1, length - 2, 2))
        return generate(position, operators, numbers) + op + generate(length - position - 1, operators, numbers)


# def sign(val):
#     return np.array([1]) if val>0 else np.array([0])

def embed(expression, embedding):
    val = numexpr.evaluate(expression)
    # label = torch.from_numpy(sign(val))
    feat = torch.from_numpy(
        np.array([[1 if embedding[i] == sym else 0 for i in range(len(embedding))] for sym in expression]))
    return feat.long(), np.array([val/100])


# def data(sequence, embedding):
#   features = []
#   labels = []
#   for seq in sequence:
#     f, l = embed(seq, embedding)
#     features.append(f)
#     labels.append(l)
#   features = torch.stack(features, 1)
#   labels = torch.cat(labels)
#   return features.T, labels

def list_of_tuples(sequence, embedding):
    list_of_tuples = []
    for seq in sequence:
        f, l = embed(seq, embedding)
        list_of_tuples.append((f, l))
    return list_of_tuples


def get_x_y_list(NUM_EXAMPLES):
    LENGTH_EXPRESSION = 31
    operators = np.array(['+', '-', '*', '/'])
    numbers = np.array(range(1, 10)).astype('str_')
    id = torch.eye(15)
    # embedding = {'1': id[2],
    #            '2': id[3],
    #            '3': id[4],
    #            '4': id[5],
    #            '5': id[6],
    #            '6': id[7],
    #            '7': id[8],
    #            '8': id[9],
    #            '9': id[10],
    #            '+': id[11],
    #            '-': id[12],
    #            '*': id[13],
    #            '/': id[14],
    #            '(': id[0],
    #            ')': id[1], }
    embedding = [')', '(', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '/']
    sequence = []
    for _ in range(NUM_EXAMPLES):
        ex = generate(LENGTH_EXPRESSION, operators, numbers)
        try:
            val = numexpr.evaluate(ex)
            if np.abs(val) > 100:
                continue
        except:
            continue
        sequence.append(ex)
    return list_of_tuples(sequence, embedding)


def generate_training_data():
    return get_x_y_list(500000)


def generate_test_data():
    return get_x_y_list(5000)


if __name__ == "__main__":
    data = get_x_y_list(500)
    mydataloader = DataLoader(data, batch_size=20)
    for X, y in mydataloader:
        print(X[0])
        print(y[0])
        break