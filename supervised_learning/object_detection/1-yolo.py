#!/usr/bin/env python3
'''initialize'''
import tensorflow
import keras
import numpy as np
import math

class Yolo:
    '''class'''
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        '''instance attr'''
        self.model = tensorflow.keras.models.load_model(model_path)
        self.class_names = [line.rstrip('\n') for line in
                            open(classes_path, "r")]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
    def process_outputs(self, outputs, image_size):
        boxes = [x[..., :4] for x in outputs]
        box_confidences = [1/(1+np.exp(- x[:, :, :, 4])) for x in outputs]
        box_class_probs = [x[:, :, :, 5:] for x in outputs]
        return (boxes, box_confidences, box_class_probs)
