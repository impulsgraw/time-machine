import os
import helper.common as paths
import logging as log
from openvino.inference_engine import IECore
import numpy as np
import cv2 as cv


class SuperresNetwork:
    MODEL_XML = os.path.join(paths.MODELS_PATH,
                             "intel/single-image-super-resolution-1032/FP32/single-image-super-resolution-1032.xml")
    MODEL_BIN = os.path.join(paths.MODELS_PATH,
                             "intel/single-image-super-resolution-1032/FP32/single-image-super-resolution-1032.bin")
    DEFAULT_OPTIONS = {
        "device": "CPU",
    }

    def __init__(self, options=None):
        options = {**SuperresNetwork.DEFAULT_OPTIONS, **(options if type(options) == dict else {})}

        log.debug("Load network")
        ie = IECore()
        self.load_net = ie.read_network(self.MODEL_XML, self.MODEL_BIN)
        self.load_net.batch_size = 1
        self.exec_net = ie.load_network(network=self.load_net, device_name=options.device)

        #  The model accepts two inputs:
        #  1: The image to be resized (3x270x480) [CxHxW]
        #  2: (optional, deprecated) The bicubic interpolation of that same image (3x1080x1920)
        #assert len(self.load_net.inputs) == 2, "Expected number of inputs is 2"
        assert len(self.load_net.inputs) == 1, "Expected number of inputs is 1"
        input_iterator = iter(self.load_net.inputs)
        self.input_blob_original = next(input_iterator)
        self.input_shape_original = self.load_net.inputs[self.input_blob_original].shape
        assert self.input_shape_original == [1, 3, 270, 480], "Input 1 does not match network shape"
        #self.input_blob_interpolated = next(input_iterator)
        #self.input_shape_interpolated = self.load_net.inputs[self.input_blob_interpolated].shape
        #assert self.input_shape_interpolated == [1, 3, 1080, 1920], "Input 2 does not match network shape"

        assert len(self.load_net.outputs) == 1, "Expected number of outputs is 1"
        self.output_blob = next(iter(self.load_net.outputs))
        self.output_shape = self.load_net.outputs[self.output_blob].shape
        assert self.output_shape == [1, 3, 1080, 1920], "Output does not match network shape"

    def input_image_size_original(self):
        _, _, h, w = self.input_shape_original
        return w, h


    #def input_image_size_interpolated(self):
     #   _, _, h, w = self.input_shape_interpolated
      #  return w, h

    #  Convention unknown
    def input_image_size(self):
        return self.input_image_size_original()


    def output_image_size(self):
        _, _, h, w = self.ouput_shape
        return w, h


    def superres(self, in_mat=None):
        w_in, h_in = self.input_image_size()
        assert in_mat.shape == (h_in, w_in, 3), f"""Incompatible dimensions of the input image matrix,
expected ${w_in}x${h_in}, got ${in_mat.shape[1]}x${in_mat.shape[0]}"""

        log.debug("Preprocessing frame")
        img_interpolated = cv.resize(in_mat.copy(), (1920, 1080))


