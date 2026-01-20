#!/usr/bin/env python3
'''
module that provide a function to add 2 matrices element-wise
'''


def add_matrices2D(mat1, mat2):
    '''
    args:
        mat1 - matrix of int/floats
        mat2 - matrix of int/floats
    return:
        new matrice with sum of matrices element-wise
    '''
    if len(mat1[0]) == len(mat2[0]):
        arr = []
        for i in range(len(mat1)):
            sub_arr = []
            for j in range(len(mat1[0])):
                sub_arr.append(mat1[i][j] + mat2[i][j])
            arr.append(sub_arr)
        return arr
    return None
