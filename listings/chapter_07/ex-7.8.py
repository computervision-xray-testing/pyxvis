# Pre-trained model
python3 predict_yolo3.py -c config_full_yolo3_infer.json -i input_path/folder -o save/folder/detection

# Training
python3 train_yolo3.py -c config_full_yolo3.json

# Testing
python3 predict_yolo3.py -c config_full_yolo3.json -i input_path/folder -o save/folder/detection

# Evaluation
python3 evaluate_yolo3.py -c config_full_yolo3.json 