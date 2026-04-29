#!/usr/bin/env python3
'''initialize'''

class Yolo:
  '''class'''
  def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
    '''instance attr'''
    self.model = model_path
    self.class_names = classes_path
    self.class_t = class_t
    self.nms_t = nms_t
    self.anchors= anchors
