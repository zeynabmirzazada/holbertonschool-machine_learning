#!/usr/bin/env python3
'''
function that returns the transpose of a 2D matrix, matrix
'''
def matrix_transpose(matrix):
    '''
    You must return a new matrix
    You can assume that matrix is never empty
    You can assume all elements in the same dimension are of same type/shape
    '''
    new=[]
    for i in range(len(matrix[0])):
        sub_new=[]
        for k in range(len(matrix)):
            sub_new.append(matrix[k][i])
        new.append(sub_new)
    return new
