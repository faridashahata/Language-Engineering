# Configurations for the code

# Data paths
TRAIN_DATA_PATH = "../data/train.jsonl"
TEST_DATA_PATH = "../data/test.jsonl"
VAL_DATA_PATH = "../data/validation.jsonl"

# Model configuration
MODEL_NAME = 't5-base'
BATCH_SIZE = 4
THRESHOLD = 350

# Training configuration
EPOCHS = 15
LEARNING_RATE = 1e-5
WARMUP_STEPS = 0
CHECKPOINT_INTERVAL = 1

# Early stopping configuration
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_MIN_DELTA = 10

# Checkpoint and save paths
CHECKPOINT_DIR = "./checkpoints"
TRAIN_STATS_PATH = "./checkpoints/train_stats.json"
VAL_STATS_PATH = "./checkpoints/val_stats.json"
MODEL_SAVE_DIR = f"./model_save_{MODEL_NAME}_{THRESHOLD}_{BATCH_SIZE}"