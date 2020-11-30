# Pre-trained model
python3 predict_yolo2.py -c config_full_yolo2_infer.json -i input_path/folder -o save/folder/detection

# Training
python3 train_yolo2.py -c config_full_yolo2.json

# Testing
python3 predict_yolo2.py -c config_full_yolo2.json -i input_path/folder -o save/folder/detection

# Evaluation
python3 evaluate_yolo2.py -c config_full_yolo2.json 