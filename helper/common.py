import os
import sys
from argparse import ArgumentParser

DATA_PATH = os.path.join(os.path.dirname(sys.argv[0]), "data")
MODELS_PATH = os.path.join(DATA_PATH, "models")


def build_arg():
    parser = ArgumentParser()
    parser.add_argument("-i", "--image_file", help="an input image file", required=True, type=str)

    outputGroup = parser.add_mutually_exclusive_group()
    outputGroup.required = True
    outputGroup.add_argument("-o", "--output_file", help="an output image file", type=str)
    outputGroup.add_argument("-s", "--show", help="display an output image on the screen instead of saving into file",
                             action="store_true", default=False)

    parser.add_argument("-d", "--device",
                        help="target device for infer: CPU, GPU, FPGA, HDDL or MYRIAD; defaults to CPU",
                        default="CPU", type=str)

    parser.add_argument("-v", "--verbose", help="enable display of processing logs",
                        action="store_true", default=False)

    parser.add_argument("-s", "--smoothing", help="use smoothing filter",
                        action="store_true", default=False)

    return parser
