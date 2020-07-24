import os
import helper.common as paths
import logging as log
from openvino.inference_engine import IECore
import numpy as np
import cv2 as cv


class InpaintNetwork:
    MODEL_XML = os.path.join(paths.MODELS_PATH, "public/gmcnn-places2-tf/frozen_model.xml")
    MODEL_BIN = os.path.join(paths.MODELS_PATH, "public/gmcnn-places2-tf/frozen_model.bin")
    DEFAULT_OPTIONS = {
        "device": "CPU",
    }

    def __init__(self, options=None):
        options = {**InpaintNetwork.DEFAULT_OPTIONS, **(options if type(options) == dict else {})}

        log.debug("Load network")
        ie = IECore()
        self.load_net = ie.read_network(InpaintNetwork.MODEL_XML, InpaintNetwork.MODEL_BIN)

        assert len(self.load_net.input_info) == 2, "Expected 2 input blob"
        assert len(self.load_net.outputs) == 1, "Expected 1 output blobs"

        self._input_layer_names = sorted(self.load_net.input_info)
        self._output_layer_name = next(iter(self.load_net.outputs))

        self._exec_model = ie.load_network(network=self.load_net, device_name=options["device"])
        self.infer_time = -1

        _, channels, input_height, input_width = self.load_net.input_info[self._input_layer_names[0]].input_data.shape
        assert channels == 3, "Expected 3-channel input"

        _, channels, mask_height, mask_width = self.load_net.input_info[self._input_layer_names[1]].input_data.shape
        assert channels == 1, "Expected 1-channel input"

        assert mask_height == input_height and mask_width == input_width, "Mask size expected to be equal to image size"
        self.input_height = input_height
        self.input_width = input_width

        self.parts = options["parts"]
        self.max_brush_width = options["max_brush_width"]
        self.max_length = options["max_length"]
        self.max_vertex = options["max_vertex"]

    def input_image_size(self):
        return self.input_height, self.input_width

    @staticmethod
    def _free_form_mask(mask, max_vertex, max_length, max_brush_width, h, w, max_angle=360):
        num_strokes = np.random.randint(max_vertex)
        start_y = np.random.randint(h)
        start_x = np.random.randint(w)
        brush_width = 0

        for i in range(num_strokes):
            angle = np.random.random() * np.deg2rad(max_angle)
            if i % 2 == 0:
                angle = 2 * np.pi - angle
            length = np.random.randint(max_length + 1)
            brush_width = np.random.randint(10, max_brush_width + 1) // 2 * 2
            next_y = start_y + length * np.cos(angle)
            next_x = start_x + length * np.sin(angle)

            next_y = np.clip(next_y, 0, h - 1).astype(np.int)
            next_x = np.clip(next_x, 0, w - 1).astype(np.int)
            cv.line(mask, (start_y, start_x), (next_y, next_x), 1, brush_width)
            cv.circle(mask, (start_y, start_x), brush_width // 2, 1)

            start_y, start_x = next_y, next_x
        return mask

    def preprocess(self, image):
        image = cv.resize(image, (self.input_width, self.input_height))
        mask = np.zeros((self.input_height, self.input_width, 1), dtype=np.float32)

        for _ in range(self.parts):
            mask = self._free_form_mask(mask, self.max_vertex, self.max_length, self.max_brush_width,
                                        self.input_height, self.input_width)

        image = image * (1 - mask) + 255 * mask
        return image, mask

    def infer(self, image, mask):
        t0 = cv.getTickCount()
        output = self._exec_model.infer(inputs={self._input_layer_names[0]: image, self._input_layer_names[1]: mask})
        self.infer_time = (cv.getTickCount() - t0) / cv.getTickFrequency()
        return output[self._output_layer_name]

    def inpaint(self, in_mat=None):
        masked_image, mask = self.preprocess(in_mat)
        image = np.transpose(masked_image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)
        output = self.infer(image, mask)

        output = np.transpose(output, (0, 2, 3, 1)).astype(np.uint8)
        #output[0] = cv.cvtColor(output[0], cv.COLOR_RGB2BGR)
        # masked_image = masked_image.astype(np.uint8)
        return output[0]
