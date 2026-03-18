import json
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
LABEL_DICHOTOMIZES = ["bias_center", "bias_right", "bias_left"]
TOKENIZER_MAX_LENGTH = 1e5 # Absurdly high so that the number of max_tokenizer is set to the model's limit
MACHINE_BATCH_SIZE = 4
N_EPOCH = 4
WEIGHT_DECAYS = [0.01, 0.05, 0.1]
LEARNING_RATES = [5e-4, 1e-4, 5e-5, 1e-5]

# model = "answerdotai/ModernBERT-base"
# model = "google-bert/bert-base-uncased"

logger = CustomLogger("./custom_logs")

for weight_decay in WEIGHT_DECAYS:
    for learning_rate in LEARNING_RATES:
        for label_dich in LABEL_DICHOTOMIZES: 
            logger(f"weight_decay: {weight_decay} | learning_rate: {learning_rate} | Label: {label_dich}", type="CASE-STUDY")
            # LOOP =========================================================================
            DH, pipe = None, None
            try : 
                # Step 1 - Training
                DH = DataHandler(
                    filename = "./data/ideology_news-stratified_year_balanced.csv", 
                    text_column = "content", 
                    label_column = label_dich,
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
                    weight_decay     = weight_decay, 
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
                with open(f"{output_dir}/scores.json", "w") as file:
                    json.dump(score_f1, file, ensure_ascii=True, indent=4)
                
                best_epoch = (
                    sorted(
                        score_f1.items(),
                        key = lambda item : item[1], # use the score
                        reverse = True
                    )
                    [0] # First item
                    [0] # epoch number
                )
                logger(f"After testing, best epoch : {best_epoch}", skip_line="before")
            
                # Step 3 - Saving embeddings
                for epoch in tqdm(range(1, N_EPOCH + 1) , desc="Exporting"):
                    export_routine = ExportEmbeddingsForOneEpoch(
                            foldername_model=output_dir,
                            foldername_data=f"./data/dataset-dict-for-inference-{label_dich}",
                            epoch = epoch,
                            logger = logger,
                            batch_size = MACHINE_BATCH_SIZE * 2
                        )
                    if epoch == best_epoch:
                        export_routine.routine(delete_files_after_routine = True)
                    else: 
                        export_routine.delete_files()
                    del export_routine
                    clean()
            
            except Exception as e:
                print("#" * 100)
                print(f"ERROR during {model} - {learning_rate}")
                print(e)
                print("#" * 100)
            
            finally : 
                del DH, pipe
                clean()
        logger.notify_when_done(f"Routine ideology_news https://projet-french-media-database-378817-0.lab.groupe-genes.fr/lab/tree/AMORIN/llm-hacking/pipeline-ideology_news.py")