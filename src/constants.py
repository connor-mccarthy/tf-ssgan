import os

PROJECT_ROOT = os.path.realpath(os.path.dirname(os.path.dirname(__file__)))
SAVED_MODEL_DIR = os.path.join(PROJECT_ROOT, "saved_models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

BERT_POOLED_OUTPUT_DIMS = 512
RANDOM_SEED = 0

PREPROCESSING_MODEL = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
SMALL_BERT = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/2"
