import os
import helper.common as paths


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

        raise NotImplementedError("Method not implemented yet")

    def input_image_size(self):
        raise NotImplementedError("Method not implemented yet")

    def output_image_size(self):
        raise NotImplementedError("Method not implemented yet")

    def superres(self, in_mat=None):
        raise NotImplementedError("Method not implemented yet")