BATCH_SIZE = 64
TRAIN_SAMPLES = BATCH_SIZE * 10
VAL_SAMPLES = BATCH_SIZE * 20
TEST_SAMPLES = BATCH_SIZE * 50
ADJUSTED_TOTAL_SAMPLES = TRAIN_SAMPLES + TEST_SAMPLES + VAL_SAMPLES
AG_NEWS_NUM_LABELED_CLASSES = 4

BERT_POOLED_OUTPUT_DIMS = 512
