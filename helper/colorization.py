import os
import helper.common as paths


class ColorizationNetwork:
    MODEL_XML = os.path.join(paths.MODELS_PATH, "public/coloriation-v2/colorization-v2.xml")
    MODEL_NPY = os.path.join(paths.MODELS_PATH, "public/coloriation-v2/colorization-v2.npy")

    @staticmethod
    def colorize(in_mat=None, out_mat=None):
        print("Colorize")
