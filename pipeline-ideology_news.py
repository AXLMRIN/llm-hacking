from toolbox import (
    DataHandler, 
    CustomTransformersPipeline, 
    TestAllEpochs, 
    ExportEmbeddingsForAllEpochs,
    CustomLogger,
    clean
)
from torch import set_float32_matmul_precision
set_float32_matmul_precision('high')

TEST_MODE = True
TOKENIZER_MAX_LENGTH = 1e5 # Absurdly high so that the number of max_tokenizer is set to the model's limit
MACHINE_BATCH_SIZE = 8

model = "answerdotai/ModernBERT-base"
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
        num_train_epochs = 4,
        
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
    (       
        TestAllEpochs(
            output_dir, 
            logger = logger, 
            batch_size = MACHINE_BATCH_SIZE
        )
        .routine(
            foldername_data = "./data/DataHandlerForInference",
            filename_scores = "./results/ideology_news/scores.csv",
            foldername_predictions = "./results/ideology_news/prediction"
        )
    )

    # Step 3 - Saving embeddings
    (
        ExportEmbeddingsForAllEpochs(
            output_dir,
            logger = logger, 
            batch_size=MACHINE_BATCH_SIZE
        )
        .routine(
            foldername_data = "./data/DataHandlerForInference",
            delete_files_after_routine=True
        )
    )
    logger.notify_when_done("Routine S12-ideology_news")
except Exception as e:
    print("#" * 100)
    print(f"ERROR during {model} - {learning_rate}")
    print(e)
    print("#" * 100)

finally : 
    del DH, pipe
    clean()

