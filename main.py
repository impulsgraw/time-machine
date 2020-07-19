from helper import colorization, cv_tools, superres, common


def main():
    cli_args = common.build_arg().parse_args()
    # the suggested flow is as follows:
    # 1) parse an input image from args
    # 2) [cv_tools] create a cv.Mat out of input image
    # 3) [cv_tools] preprocess image: inpaint
    # 4) [cv_tools] preprocess image: resize to [colorization] input
    # 5) [colorization] proceed with colorization
    # 6) [cv_tools] resize (interpolate) to [superres] input
    # 7) [superres] proceed with super-resolution
    # 8) [cv_tools] resize (interpolate) to the former image input
    # 9) output an image somewhere

    # ... input ...

    cv_tools.init_image()
    cv_tools.inpaint()
    cv_tools.scale()

    colorization.ColorizationNetwork.colorize()

    cv_tools.scale()

    superres.SuperresNetwork.superres()

    cv_tools.scale()

    # ... output ...


if __name__ == "__main__":
    main()
