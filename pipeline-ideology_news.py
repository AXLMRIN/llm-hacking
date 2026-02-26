from tqdm import tqdm

from toolbox import (
    DataHandler, 
    CustomTransformersPipeline, 
    TestOneEpoch, 
    ExportEmbeddingsForOneEpoch,
    CustomLogger,
    clean
)
from torch import set_float32_matmul_precision
set_float32_matmul_precision('high')

TEST_MODE = True
TOKENIZER_MAX_LENGTH = 1e5 # Absurdly high so that the number of max_tokenizer is set to the model's limit
MACHINE_BATCH_SIZE = 2
N_EPOCH = 4

# model = "answerdotai/ModernBERT-base"
model = "google-bert/bert-base-uncased"
learning_rate = 5e-4

logger = CustomLogger("./custom_logs")
# LOOP =========================================================================
DH, pipe = None, None
try : 
    # Step 1 - Training
    DH = DataHandler(
        filename = "./data/ideology_news-stratified_year_balanced.csv",   # UPDATE
        text_column = "content", 
        label_column = "bias_text",
        id_column = "ID",
        logger = logger, 
    )
    DH.routine()
    
    pipe = CustomTransformersPipeline(
        data             = DH, 
        model_name       = model,
        num_train_epochs = N_EPOCH,
        
        total_batch_size = 64,
        learning_rate    = learning_rate,
        weight_decay     = 0.01, 
        warmup_ratio     = 0.1, 

        
        batch_size_device = MACHINE_BATCH_SIZE,

        tokenizer_max_length = TOKENIZER_MAX_LENGTH, 
        logger           = logger,
        disable_tqdm     = False, 
    )
    pipe.routine(debug_mode = TEST_MODE)

    output_dir = pipe.output_dir
    del DH, pipe
    clean()
    DH, pipe = (None, None)

    # Step 2 - Testing
    score_f1 = {}
    for epoch in tqdm(range(1, N_EPOCH + 1) , desc="Testing"):
        score, _ = (
            TestOneEpoch(
                foldername_model = output_dir, 
                epoch = epoch, 
                logger = logger, 
                batch_size = MACHINE_BATCH_SIZE * 2
            )
            .routine()
        )
        score_f1[epoch] = score["score"]
    
    best_epoch = (
        sorted(
            score_f1.items(),
            key = lambda item : item[1], # use the score
            reverse = True
        )
        [0] # First item
        [0] # epoch number
    )
    logger(f"After testing, best epoch : {best_epoch}")

    # Step 3 - Saving embeddings
    (
        ExportEmbeddingsForOneEpoch(
            foldername_model=output_dir,
            epoch = best_epoch,
            logger = logger,
            # foldername_data=
            batch_size = MACHINE_BATCH_SIZE * 2
        )
        .routine(delete_files_after_routine=True)
    )
    # (
    #     ExportEmbeddingsForAllEpochs(
    #         output_dir,
    #         logger = logger, 
    #         batch_size=MACHINE_BATCH_SIZE * 2
    #     )
    #     .routine(
    #         foldername_data = "./data/DataHandlerForInference",
    #         delete_files_after_routine=True
    #     )
    # )
    logger.notify_when_done("Routine S12-ideology_news")
except Exception as e:
    print("#" * 100)
    print(f"ERROR during {model} - {learning_rate}")
    print(e)
    print("#" * 100)

finally : 
    del DH, pipe
    clean()

