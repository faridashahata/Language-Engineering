# Hyperparameters
BATCH_SIZE = 2
LEARNING_RATE = 3e-5
EPOCHS = 3
THRESHOLD = 350

TRAIN_DATA_PATH = "../data/train_example.jsonl"
VAL_DATA_PATH = "../data/validation_example.jsonl"
TEST_DATA_PATH = "../data/test_example.jsonl"
MODEL_NAME = "facebook/bart-base"
MODEL_DIR = "./model_save_bart"
MODEL_FILE_NAME = "model_bart.pt"
OPTIMIZER_FILE_NAME = "optimizer.pt"

#Test configuration
MODEL_TO_TEST = "model_save_bart/model_bart.pt"