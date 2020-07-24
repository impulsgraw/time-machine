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
    outputGroup.add_argument("-o", "--output_dir", help="an output image folder path", type=str)
    outputGroup.add_argument("-s", "--show", help="display an output image on the screen instead of saving into file",
                             action="store_true", default=False)

    parser.add_argument("-d", "--device",
                        help="target device for infer: CPU, GPU, FPGA, HDDL or MYRIAD; defaults to CPU",
                        default="CPU", type=str)

    parser.add_argument("-v", "--verbose", help="enable display of processing logs",
                        action="store_true", default=False)

    parser.add_argument("-S", "--smoothing", help="use smoothing filter; not used by default",
                        action="store_true", default=False)

    inpaintNetworkGroup = parser.add_argument_group("Inpaint network")
    inpaintNetworkGroup.add_argument("--inpaint", help="use inpaint network; not used by default",
                                     action="store_true", default=False)
    inpaintNetworkGroup.add_argument("-p", "--parts", help="optional: number of parts to draw mask",
                                     default=8, type=int)
    inpaintNetworkGroup.add_argument("-mbw", "--max_brush_width", help="optional: max width of brush to draw mask",
                                     default=24, type=int)
    inpaintNetworkGroup.add_argument("-ml", "--max_length", help="optional: max strokes length to draw mask",
                                     default=100, type=int)
    inpaintNetworkGroup.add_argument("-mv", "--max_vertex", help="optional: max number of vertex to draw mask",
                                     default=20, type=int)

    return parser
