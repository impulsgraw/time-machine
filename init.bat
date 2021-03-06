@echo off

IF NOT DEFINED INTEL_OPENVINO_DIR (
	echo Please run OpenVINO setupvars.bat first
	exit /b
)

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing venv...
python -m pip install venv

echo Initializing venv...
python -m venv .\venv

echo Activating venv...
call .\venv\Scripts\activate.bat

echo Installing required packages...
python -m pip install -r requirements.txt

IF NOT EXIST .\data\models mkdir .\data\models

echo Running python scripts...
REM Windows is perfectly capable of ingesting paths with forward slashes
python "%INTEL_OPENVINO_DIR%/deployment_tools/tools/model_downloader/downloader.py" --name colorization-v2 -o ./data/models/
python "%INTEL_OPENVINO_DIR%/deployment_tools/model_optimizer/mo.py" --input_model ./data/models/public/colorization-v2/colorization-v2.caffemodel --input_proto ./data/models/public/colorization-v2/colorization-v2.prototxt -o ./data/models/public/colorization-v2/ --input_shape [1,1,224,224] --input=data_l --mean_values=data_l[50] --output=class8_313_rh
python "%INTEL_OPENVINO_DIR%/deployment_tools/tools/model_downloader/downloader.py" --name single-image-super-resolution-1032 -o ./data/models/
echo Done
exit /b
