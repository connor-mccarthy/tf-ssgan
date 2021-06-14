import os

N_LABELED_CLASSES = 4
BATCH_SIZE = 128
BERT_POOLED_OUTPUT_DIMS = 512
TRAIN_SAMPLES = 1024
TEST_SAMPLES = 1024
VAL_SAMPLES = 1024
LATENT_VECTOR_DIM = 100
RANDOM_SEED = 0
EPSILON = 1e-10
UNLABELED_TO_LABELED_RATIO = 100

PROJECT_ROOT = os.path.dirname(os.path.realpath(os.path.dirname(__file__)))
IMAGE_DIR = os.path.join(PROJECT_ROOT, "architecture_images/")
SAVED_MODELS_DIR = os.path.join(PROJECT_ROOT, "saved_models/")

PREPROCESSING_MODEL = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
SMALL_BERT = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/2"
