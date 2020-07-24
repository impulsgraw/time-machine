from helper import colorization, cv_tools, superres, inpaint, common
import logging as log
import sys
import cv2 as cv
import os


def main():
    # grab cli arguments
    cli_args = common.build_arg().parse_args()

    # setup logging level
    log.basicConfig(format="[ %(levelname)s ] %(message)s",
                    level=log.INFO if not cli_args.verbose else log.DEBUG, stream=sys.stdout)

    # parse an input image from args
    # create a cv.Mat out of input image
    original_img = cv.imread(cli_args.image_file)
    img_mat = cv_tools.init_image(cli_args.image_file, {"smoothing": cli_args.smoothing})
    original_constraints = img_mat.shape[:2]


    ##### INPAINT PART #####
    if cli_args.inpaint:
        inpaint_network = inpaint.InpaintNetwork({
            "device": cli_args.device,
            "parts": cli_args.parts,
            "max_brush_width": cli_args.max_brush_width,
            "max_length": cli_args.max_length,
            "max_vertex": cli_args.max_vertex
        })
        # preprocess image: resize to [colorization] input
        img_mat = cv_tools.scale(img_mat, *inpaint_network.input_image_size())
        # proceed with colorization
        img_mat = inpaint_network.inpaint(img_mat)

    img_mat = cv.cvtColor(img_mat, cv.COLOR_BGR2GRAY)

    ##### COLORIZATION PART #####
    colorization_network = colorization.ColorizationNetwork({
        "device": cli_args.device
    })

    # preprocess image: resize to [colorization] input
    img_mat = cv_tools.scale(img_mat, *colorization_network.input_image_size())
    # proceed with colorization
    img_mat = colorization_network.colorize(img_mat)


    ##### SUPER-RESOLUTION PART #####
    superres_network = superres.SuperresNetwork({
        "device": cli_args.device
    })

    # resize (interpolate) to [superres] input
    img_mat = cv_tools.scale(img_mat, *superres_network.input_image_size())
    # proceed with super-resolution
    img_mat = superres_network.superres(img_mat)


    ##### MISCELLANEOUS #####
    # resize (interpolate) to the former image input
    img_mat = cv_tools.scale(img_mat, *original_constraints)

    # output an image somewhere
    if cli_args.output_dir:
        cv_tools.output((original_img, img_mat * 255), os.path.join(cli_args.output_dir, os.path.split(cli_args.image_file)[1]))

    if cli_args.show:
        cv_tools.show((original_img / 255, img_mat))


if __name__ == "__main__":
    main()
