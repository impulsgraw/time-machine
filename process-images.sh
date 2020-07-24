mkdir -p ./data/test_output/inpaint

for file in ./data/test_images/real_photos/*
do
  python main.py --image_file $file -S -o ./data/test_output/inpaint --inpaint
done