#!/usr/bin/env python3
'''
module that provide a function to add 2 arrays element-wise
'''


def add_arrays(arr1, arr2):
    '''
    args:
        arr1 - list of int/floats
        arr2 - list oof int/floats
    return:
        new list with sum of arrays element-wise
    '''
    if len(arr1) == len(arr2):
        arr = []
        for i in range(len(arr1)):
            arr.append(arr1[i] + arr2[i])
        return arr
    return None
