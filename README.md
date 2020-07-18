# Time Machine
An application for colorizing and recovering old black&amp;white photos and videos

## Project startup
```shell script
python3 ./main.py --image_file test-image.jpg --output recovered-image.jpg
```

## Local project dev environment setup
For the first run one needs to:
0. Install [Intel OpenVINO](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/choose-download.html) with _Python3_
1. Run `(OPEN_VINO INSTALL PATH)/bin/setupvars.sh`
2. Run `init.sh` shell script (for Linux/MacOS), which does:
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
- [ ] Command line args parsing + help output
- [ ] Colorization network execution
- [ ] Super-resolution network execution
- [ ] cv_tools implementation