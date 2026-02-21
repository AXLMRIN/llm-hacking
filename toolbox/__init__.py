from .train_embedding_models import *
from .test_embedding_models import *
from .optimize_classifiers import *
from .save_embeddings import *
from .visualise_results import *
from .general import *
from .CustomLogger import CustomLogger

import os 
required_folders = ["data", "models", "custom_logs", "figures", "results"]
for f in required_folders:
    if not os.path.isdir(f"./{f}"):
        os.makedirs(f"./{f}")
