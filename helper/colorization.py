import os
import helper.common as paths
import logging as log
from openvino.inference_engine import IECore
import numpy as np
import cv2 as cv


class ColorizationNetwork:
    MODEL_XML = os.path.join(paths.MODELS_PATH, "public/colorization-v2/colorization-v2.xml")
    MODEL_BIN = os.path.join(paths.MODELS_PATH, "public/colorization-v2/colorization-v2.bin")
    MODEL_COEFS = os.path.join(paths.MODELS_PATH, "public/colorization-v2/colorization-v2.npy")
    DEFAULT_OPTIONS = {
        "device": "CPU",
    }

    def __init__(self, options=None):
        options = {**ColorizationNetwork.DEFAULT_OPTIONS, **(options if type(options) == dict else {})}

        log.debug("Load network")
        ie = IECore()
        self.load_net = ie.read_network(ColorizationNetwork.MODEL_XML, ColorizationNetwork.MODEL_BIN)
        self.load_net.batch_size = 1
        self.exec_net = ie.load_network(network=self.load_net, device_name=options["device"])

        assert len(self.load_net.input_info) == 1, "Expected number of inputs is equal 1"
        self.input_blob = next(iter(self.load_net.input_info))
        self.input_shape = self.load_net.input_info[self.input_blob].input_data.shape
        assert self.input_shape[1] == 1, "Expected model input shape with 1 channel"

        assert len(self.load_net.outputs) == 1, "Expected number of outputs is equal 1"
        self.output_blob = next(iter(self.load_net.outputs))
        self.output_shape = self.load_net.outputs[self.output_blob].shape
        assert self.output_shape == [1, 313, 56, 56], "Shape of outputs does not match network shape outputs"

        self.color_coeff = np.load(ColorizationNetwork.MODEL_COEFS).astype(np.float32)
        assert self.color_coeff.shape == (313, 2), "Current shape of color coefficients does not match required shape"

    def input_image_size(self):
        _, _, h_in, w_in = self.input_shape
        return h_in, w_in

    def output_image_size(self):
        _, _, h_out, w_out = self.output_shape
        return h_out, w_out

    def colorize(self, in_mat=None):
        h_in, w_in = self.input_image_size()

        assert in_mat.shape == (h_in, w_in), f"""Incompatible dimensions of 
            input image matrix, expected {(h_in, w_in)} but got {in_mat.shape}"""

        log.debug("Preprocessing frame")
        frame = cv.cvtColor(in_mat, cv.COLOR_GRAY2RGB)

        img_rgb = frame.astype(np.float32) / 255
        img_lab = cv.cvtColor(img_rgb, cv.COLOR_RGB2Lab)
        img_l_rs = cv.resize(img_lab.copy(), (w_in, h_in))[:, :, 0]

        log.debug("Network inference")
        res = self.exec_net.infer(inputs={self.input_blob: [img_l_rs]})

        update_res = (res[self.output_blob] * self.color_coeff.transpose()[:, :, np.newaxis, np.newaxis]).sum(1)

        log.debug("Get results")
        out = update_res.transpose(1, 2, 0)
        out = cv.resize(out, (w_in, h_in))
        img_lab_out = np.concatenate((img_lab[:, :, 0][:, :, np.newaxis], out), axis=2)
        img_bgr_out = np.clip(cv.cvtColor(img_lab_out, cv.COLOR_Lab2BGR), 0, 1)

        return img_bgr_out
