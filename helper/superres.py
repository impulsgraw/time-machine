import os
import helper.common as paths


class SuperresNetwork:
    MODEL_PATH = os.path.join(paths.DATA_PATH, "intel/single-image-super-resolution-1032/FP32/single-image-super-resolution-1032.xml")

    @staticmethod
    def superres(in_mat=None, out_mat=None):
        print("Superres")