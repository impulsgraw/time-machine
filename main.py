from helper import colorization, cv_tools, superres, common
import logging as log
import sys


def main():
    # grab cli arguments
    cli_args = common.build_arg().parse_args()

    # setup logging level
    log.basicConfig(format="[ %(levelname)s ] %(message)s",
                    level=log.INFO if not cli_args.verbose else log.DEBUG, stream=sys.stdout)

    # parse an input image from args
    # create a cv.Mat out of input image
    img_mat = cv_tools.init_image(cli_args.image_file, {"smoothing": cli_args.smoothing})
    original_constraints = img_mat.shape[:2]

    # preprocess image: inpaint
    # img_mat = cv_tools.inpaint(img_mat)

    colorization_network = colorization.ColorizationNetwork({
        "device": cli_args.device
    })

    # preprocess image: resize to [colorization] input
    img_mat = cv_tools.scale(img_mat, *colorization_network.input_image_size())
    # proceed with colorization
    img_mat = colorization_network.colorize(img_mat)

    superres_network = superres.SuperresNetwork({
        "device": cli_args.device
    })

    # resize (interpolate) to [superres] input
    img_mat = cv_tools.scale(img_mat, *superres_network.input_image_size())
    # proceed with super-resolution
    img_mat = superres_network.superres(img_mat)

    # resize (interpolate) to the former image input
    img_mat = cv_tools.scale(img_mat, *original_constraints)

    # output an image somewhere
    if cli_args.output_file:
        cv_tools.output(img_mat, cli_args.output_file)

    if cli_args.show:
        cv_tools.show(img_mat)



if __name__ == "__main__":
    main()
