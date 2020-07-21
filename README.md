# Time Machine
An application for colorizing and recovering old black&amp;white photos and videos

## Example usage
```shell script
python3 main.py --image_file test-image.jpg --output_file recovered-image.jpg
```

Full options list is as follows:
```text
usage: main.py [-h] -i IMAGE_FILE (-o OUTPUT_FILE | -s) [-d DEVICE] [-v]

optional arguments:
  -h, --help            show this help message and exit
  -i IMAGE_FILE, --image_file IMAGE_FILE
                        an input image file
  -o OUTPUT_FILE, --output_file OUTPUT_FILE
                        an output image file
  -s, --show            display an output image on the screen instead of
                        saving into file
  -d DEVICE, --device DEVICE
                        target device for infer: CPU, GPU, FPGA, HDDL or
                        MYRIAD; defaults to CPU
  -v, --verbose         enable display of processing logs
  -s, --smoothing	use smoothing filter
```

## Local project dev environment setup
For the first run one needs to:
1. Install [Intel OpenVINO](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/choose-download.html) with _Python3_
2. Run `(OPEN_VINO INSTALL PATH)/bin/setupvars.sh`
3. Run `init.sh` shell script (for Linux/MacOS), which does:
    1. Setup a python virtualenv
    2. Install **requirements.txt**
    3. Create **./data/models** folder
    4. Download necessary models

```shell script
source ./init.sh
```

Formerly recommend to work within local virtualenv


## Project structure
* `main.py` -- main project file
* `helper/`
    * `colorization.py` -- colorization network invocation
    * `superres.py`     -- super-resolution network invocation
    * `cv_tools.py`     -- helper functions on OpenCV
    * `common.py`       -- global variable definitions
* `data/` -- generated after the init script is being executed
    * `models/` -- expected to contain models automatically downloaded by 
    **Intel OpenVINO ModelDownloader** and processed by **ModelOptimizer** down to the IR format
* `requirements.txt` -- pip virtualenv requirements
* `LICENSE` -- license information
    

## TODOs
- [ ] Create an init shell script for Windows (?)
- [x] Command line args parsing + help output
- [ ] Colorization network execution
- [ ] Super-resolution network execution
- [ ] cv_tools implementation
