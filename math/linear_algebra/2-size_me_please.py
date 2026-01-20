#!/usr/bin/env python3
'''
function that calculates the shape of a matrix
'''
def matrix_shape(matrix):
    '''
    assume all elements in the same dimension are of the same type/shape
    The shape should be returned as a list of integers
    '''
    return list(numpy.array(matrix).shape)
