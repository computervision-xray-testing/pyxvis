# Pre-trained model
python3 predict_ssd.py -c config_7_infer.json -i input_path/folder -o save/folder/detection

# Training
python3 train_ssd.py -c config_7.json

# Testing
python3 predict_ssd.py -c config_7.json -i input_path/folder -o save/folder/detection

# Evaluation
python3 evaluate_ssd.py -c config_7_infer.json 