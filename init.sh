if [ -z $INTEL_OPENVINO_DIR ]
then
  echo "Please run OpenVINO setupvars.sh first"
fi

# activate the pipenv
pip install venv
source venv/bin/activate
pip install -r requirements.txt

# generate models
if [ ! -d "./data/models" ]; then
  mkdir -p ./data/models
fi

python3 -mpip install --user -r "$INTEL_OPENVINO_DIR/deployment_tools/tools/model_downloader/requirements.in"

python3 "$INTEL_OPENVINO_DIR/deployment_tools/tools/model_downloader/downloader.py" --name colorization-v2 -o ./data/models/
python3 "$INTEL_OPENVINO_DIR/deployment_tools/model_optimizer/mo.py" --input_model ./data/models/public/colorization-v2/colorization-v2.caffemodel --input_proto ./data/models/public/colorization-v2/colorization-v2.prototxt -o ./data/models/public/colorization-v2/ --input_shape [1,1,224,224] --input=data_l --mean_values=data_l[50] --output=class8_313_rh

python3 "$INTEL_OPENVINO_DIR/deployment_tools/tools/model_downloader/downloader.py" --name single-image-super-resolution-1033 -o ./data/models/

python3 "$INTEL_OPENVINO_DIR/deployment_tools/tools/model_downloader/downloader.py" --name gmcnn-places2-tf -o ./data/models/
python3 "$INTEL_OPENVINO_DIR/deployment_tools/model_optimizer/mo.py" --input_model ./data/models/public/gmcnn-places2-tf/frozen_model.pb -o ./data/models/public/gmcnn-places2-tf/ --input_shape=[1,512,680,3],[1,512,680,1] --input=Placeholder,Placeholder_1 --output=Minimum