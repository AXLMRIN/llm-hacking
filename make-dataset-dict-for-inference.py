from toolbox import DataHandler, CustomLogger
from transformers import AutoTokenizer

model = "answerdotai/ModernBERT-base"

logger = CustomLogger("./custom_logs")

DH = DataHandler(
    filename = "./data/ideology_news-dataset_for_inference.csv",
    text_column="content",
    label_column = "bias_right",
    id_column = "ID", 
    logger = logger
)

DH.open_data()
DH.split(ratio_train=0, ratio_eval=0)
DH.debug_mode()


tokenizer = AutoTokenizer.from_pretrained(model)
tokenizing_parameters = {
    'padding' : 'max_length',
    'truncation' : True,
    'max_length' : 8192
}
DH.encode(tokenizer, tokenizing_parameters)
DH.save_all("./data")