# llm-hacking

repo for llm-hacking investigation. 

## Objectif: 

**Adapter [LLM Hacking](https://arxiv.org/pdf/2509.08825) au cas des classifieurs encodeurs**

Ça signifie:
    
- Prendre des jeux de données dans leur banque (p.9)
- Produire des labels grâce à un classifieur basé sur l'architecture encodeur
- Comparer les résultats de régressions linéaires $label_{true} \sim y$ et $label_{pred} \sim y$ (Erreur I, II, S et M)

## Protocole expérimental: 

### Selection du jeu de données et prétraitement

- Choix d'un jeu de données avec **au moins 2000 éléments** et qui **contient des métadonnées intéressantes**. _Jeux de données considérés: manifestos (Léo), ideology news (Axel), misinfo (Alexandre)._
- Création des splits (Train, Test, Inférence). `N_train` est un hyperparamètre, `N_inference` doit correspondre au nombre d'éléments annotés dans le papier LLM Hacking (p.9). 
    - Possibilité d'essayer d'avoir des overlaps entre l'ensemble d'entraînement / ensemble d'inference
- Les splits doivent être créés par tirage aléatoire _(vérifier qu'il n'y a pas de doublon)_. Le tirage peut être stratifié (à documenter).
    - _Possibilité d'étendre à un tirage non aléatoire similaire à une situation d'active learning._
 
> 🚨 **Documenter le jeux de données ainsi que les métadonnées existantes.** 🚨

### Génération des labels

- **Pour chaque label, entraînement d'un classifieur binaire**
- Ensemble d'hyperparamètres pouvant être explorés:
    - `N_train`
    - `balanced_dataset` (l'ensemble d'entraînement contient 50% de labels True, 50% de labels false)
    - `model`
    - `learning_rate`
    - `weight_decay`
    - `dropout`
    - `warmup_ratio` ???
    - Hyperparamètres supplémentaires: `pooling_method`, `sampling_method` (active ou random)
- On génère les labels sur l'ensemble d'inférence.

_Exemple de pipeline:_ `pipeline-ideology_news.py`

```python

# Choix des hyperparamètres
model = "google-bert/bert-base-uncased"
learning_rate = 5e-4
...

# Ouverture du dataset et préparation des variables
DH = DataHandler(
    filename = ...,   # UPDATE
    text_column = ..., 
    label_column = ...,
    id_column = ...,
)
DH.routine()

# Entraînement du modèle
pipe = CustomTransformersPipeline(
    data             = DH, 
    model_name       = model,
    num_train_epochs = N_EPOCH,
    
    ...
)
pipe.routine()

# Test du F1 score après N epochs
for epoch in tqdm(range(1, N_EPOCH + 1) , desc="Testing"):
    score, _ = TestOneEpoch(...).routine()
    score_f1[epoch] = score["score"]

# Choix de la meilleur epoch 🚨 Suppression de n_epoch comme hyperparamètre
best_epoch = ...

# Inférence sur l'ensemble d'inference
for epoch in tqdm(range(1, N_EPOCH + 1) , desc="Exporting"):
        export_routine = ExportEmbeddingsForOneEpoch(
                foldername_model=folder_model,
                foldername_data=folder_jeu_inference, #🚨 doit être créé en amont, voir make-dataset-dict-for-inference.py
                epoch = epoch,
                logger = logger,
            )

        # Pour la meilleure epoch, réaliser l'inférence
        if epoch == best_epoch:
            export_routine.routine(delete_files_after_routine = True)
        # Pour les autres, uniquement supprimer les fichiers lourds
        else: 
            export_routine.delete_files()
```

Sortie d'intérêt de la pipeline: 

- `output_dir/data/DataHandler_config.json` -> label dichotomizé, `N_train` 
- `output_dir/scores.json` -> F1 (macro) du modèle sur l'ensemble de test après $n$ epochs
- `output_dir/checkpoint-XX/trainer_state.json` -> loss, ...
- `output_dir/checkpoint-XX/training_args.bin` -> hyperparamètres d'entraînement
- `output_dir/embeddings/test_labels-XXXX.csv` -> csv avec l'ID, le label prédit et le label true pour chaque élément de l'ensemble d'inférence (sauvegardé pendant `ExportEmbeddingsForOneEpoch.routine()`)
- 

### Regression

- Regression des labels (prédits / gold) sur une métadonnée du jeux origine (binarisée). (ex: `sm.Logit(y = df["topic-economy"], X = df["label-centre])) 
- Sauvegarde des données de regression:
    - `Pseudo R-squared`
    - `Coef`
    - `Std err`
    - `z`
    - `pvalues`
    - `Conf Int`
    - `Log-Likelihood`
    - `LL-Null`
    - `LLR p-value`
    - `AIC`
    - `BIC`
    - `N obs`
