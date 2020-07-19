import os
import helper.common as paths


class SuperresNetwork:
    MODEL_XML = os.path.join(paths.MODELS_PATH,
                             "intel/single-image-super-resolution-1032/FP32/single-image-super-resolution-1032.xml")

    @staticmethod
    def superres(in_mat=None, out_mat=None):
        print("Superres")