echo "train wavenet"
python3 src/train.py model_name=wavenet epoch=25 batch_size=8
echo "train resnet_1"
python3 src/train.py model_name=resnet_1 epoch=25 batch_size=32
echo "train resnet_2"
python3 src/train.py model_name=resnet_2 epoch=25 batch_size=32
echo "train lstm"
python3 src/train.py model_name=lstm epoch=25 batch_size=32
