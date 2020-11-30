# Pre-trained model
python3 predict_retinanet.py -c config_resnet50_infer.json -i input_path/folder -o save/folder/detection

# Training
python3 train_retinanet.py -c config_resnet50.json

# Testing
python3 predict_retinanet.py -c config_resnet50.json -i input_path/folder -o save/folder/detection

# Evaluation
python3 evaluate_retinanet.py -c config_resnet50.json 