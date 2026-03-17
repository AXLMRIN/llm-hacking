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

- Regression d'une métadonnée du jeux origine (binarisée) sur les labels (prédits / gold). (ex: `sm.Logit(y = df["label-centre]), X = df["topic-economy"]) 
- Sauvegarde des données de regression:
    - `Pseudo R-squared`
    - `Coef`
    - `Std err`
    - `z`🟠
    - `pvalues`
    - `Conf Int`
    - `Log-Likelihood`
    - `LL-Null`
    - `LLR p-value`
    - `AIC`🟠
    - `BIC`🟠
    - `N obs`🟠
    - `N iterations`
    - 🟠: "à prioris inutile"
- Analyse des résultats:
    - Filtrer les regressions qui n'ont pas fonctionné (`res_success = res.loc['FAILED' != res['Coef']]`)
    - Créer des paires de regressions
        - grouper par dataset 
        - grouper par hypotèse (covariate explique label) — i.e. covariate x label
        - grouper par configuration (modele, learning rate etc..)
        - exemple: 
        ```python
        (
            res_success
            .groupby([
                "dataset" # ideology_news; misinfo; manifesto
                "label", "x_column",  # hypothesis
                "model", "learning_rate", "weight_decay" # configuration
            ])
        )
        ```
        - Chaque groupe devrait contenir 2 regressions, une où le label est gold-standard et un ou le label est prédit
    - Filtrer les couples de regression avec une regression manquante `valid_for_comparison = res_success.groupby([ ... ]).size() == 2`
    - Pour chaque groupe de regression évaluer la présence d'erreur
        - `error_type_1 : bool = pred_significant and not GS_significant`
        - `error_type_2 : bool = GS_significant and not pred_significant`
        - `error_type_S : bool = pred_significant and GS_significant and (GS_coef * pred_coef < 0)`
        - `error_type_M : float = pred_significant and GS_significant and (GS_coef * pred_coef < 0) * magnitude_coef`
        - _voir `analyse-regression-results.py` pour les détails_
    - Évaluer les risques d'après la définition du papier
        - Type I Risk 
        $$
        = \frac1{|T|}\sum_{t\in T}\frac1{|H_t^0|}\sum_{h\in H_t^0}\frac1{|\Phi|}\sum_{\phi \in \Phi}\mathbb 1\left[S_{h,\phi}^{LLM} = 1\right]
        $$
        code: 
        ```python
        risk = 0 
        T = 0
        PHI = len(unique_configs) # configs independantes de la tâche (dataset + labels) et des regressions
        for dataset in unique_datasets:
            T += len(unique_labels[dataset])
            for label in unique_labels[dataset]:
                H_t_0_counter = 0
                hypothesis_risk_counter = 0
                for covariate in unique_covariates[dataset]:
                    if GS_significant[dataset, label, covariate] == 0: # i.e. non signifiant
                        H_t_0_counter += 1
                        config_risk_counter = 0
                        for config in unique_configs:
                            if (pred_significant[dataset, label, covariate, config] == 1):  # i.e. signifiant
                                config_risk_counter += 1
                        hypothesis_risk_counter += config_risk_counter / PHI
                risk += hypothesis_risk_counter / H_t_0_counter
        risk_I = risk / T
        ```
        - Type II Risk 
        $$
        = \frac1{|T|}\sum_{t\in T}\frac1{|H_t^1|}\sum_{h\in H_t^1}\frac1{|\Phi|}\sum_{\phi \in \Phi}\mathbb 1\left[S_{h,\phi}^{LLM} = 0\right]
        $$
        code: 
        ```python
        risk = 0 
        T = 0
        PHI = len(unique_configs) # configs independantes de la tâche (dataset + labels) et des regressions
        for dataset in unique_datasets:
            T += len(unique_labels[dataset])
            for label in unique_labels[dataset]:
                H_t_1_counter = 0
                hypothesis_risk_counter = 0
                for covariate in unique_covariates[dataset]:
                    if GS_significant[dataset, label, covariate] == 1: # i.e. signifiant
                        H_t_1_counter += 1
                        config_risk_counter = 0
                        for config in unique_configs:
                            if (pred_significant[dataset, label, covariate, config] == 0):  # i.e. non signifiant
                                config_risk_counter += 1
                        hypothesis_risk_counter += config_risk_counter / PHI
                risk += hypothesis_risk_counter / H_t_1_counter
        risk_II = risk / T
        ```
        - Type S Risk 
        $$
        = \frac1{|T|}\sum_{t\in T}\frac1{|H_t^1|}\sum_{h\in H_t^1}\frac1{|\Phi|}\sum_{\phi \in \Phi}\mathbb 1\left[S_{h,\phi}^{LLM} = 1, sgn(\beta^{GT}_h) \neq sgn(\beta^{LLM}_{h,\phi}\right]
        $$
        code: 
        ```python
        risk = 0 
        T = 0
        PHI = len(unique_configs) # configs independantes de la tâche (dataset + labels) et des regressions
        for dataset in unique_datasets:
            T += len(unique_labels[dataset])
            for label in unique_labels[dataset]:
                H_t_1_counter = 0
                hypothesis_risk_counter = 0
                for covariate in unique_covariates[dataset]:
                    if GS_significant[dataset, label, covariate] == 1: # i.e. signifiant
                        H_t_1_counter += 1
                        config_risk_counter = 0
                        for config in unique_configs:
                            if (
                                pred_significant[dataset, label, covariate, config] == 1 # i.e. signifiant
                                and
                                coef_GT[dataset, label, covariate] * coef_pred[dataset, label, covariate, config] < 0 # i.e. pas meme signe
                            ): 
                                config_risk_counter += 1
                        hypothesis_risk_counter += config_risk_counter / PHI
                risk += hypothesis_risk_counter / H_t_1_counter
        risk_S = risk / T
        ```
        - Type M Risk 
        $$
        = \frac1{|T|}\sum_{t\in T}\frac1{|H_t^1|}\sum_{h\in H_t^1}\frac1{|\Phi|}\sum_{\phi \in \Phi}\mathbb 1\left[S_{h,\phi}^{LLM} = 1, sgn(\beta^{GT}_h) \neq sgn(\beta^{LLM}_{h,\phi}\right]
        $$
        code: 
        ```python
        risk = 0 
        T = 0
        PHI = len(unique_configs) # configs independantes de la tâche (dataset + labels) et des regressions
        for dataset in unique_datasets:
            T += len(unique_labels[dataset])
            for label in unique_labels[dataset]:
                H_t_1_counter = 0
                hypothesis_risk_counter = 0
                for covariate in unique_covariates[dataset]:
                    if GS_significant[dataset, label, covariate] == 1: # i.e. signifiant
                        H_t_1_counter += 1
                        config_risk_counter = 0
                        for config in unique_configs:
                            if (
                                pred_significant[dataset, label, covariate, config] == 1 # i.e. signifiant
                                and
                                coef_GT[dataset, label, covariate] * coef_pred[dataset, label, covariate, config] > 0 # i.e. meme signe
                            ): 
                                delta_p_pred = abs(
                                    (labels_pred[dataset, label, covariate, config] == 1).mean()
                                    - 
                                    (labels_pred[dataset, label, covariate, config] == 0).mean()
                                )
                                delta_p_GS = abs(
                                    (labels_GS[dataset, label, covariate] == 1).mean()
                                    - 
                                    (labels_GS[dataset, label, covariate] == 0).mean()
                                )
                                config_risk_counter += abs(
                                    (delta_p_pred - delta_p_GS)
                                    / 
                                    delta_p_GS
                                )
                        hypothesis_risk_counter += config_risk_counter / PHI
                risk += hypothesis_risk_counter / H_t_1_counter
        risk_M = risk / T
        ```

        - Discussion à avoir:
            - T représente l'ensemble des tâches, tandis que $H_t$ l'ensemble des hypothèses. D'après notre lecture, (Alexandre et Axel), on comprend que T revient à être la somme du nombre de tâches de classification à travers les jeux de données (i.e. $\sum_{d\in datasets}N^{labels}_d$ ) tandis que $H_t$ l'ensemble des regressions réalisé par label et par dataset (i.e. le nombre de covariates par dataset $\sum_{d\in datasets}N^{cov}_d$).
            - Les quantités "Risk" sont des moyennes, de moyennes, de moyennes, ... est-ce bien serieux?
            - aussi, il ne semble pas y avoir de contrôle sur la qualité des regressions (pas de filtre sur le F-score, ni le respect des hypothèses sur les erreurs). Est ce que le risque n'englobe pas tout un tas de regression qui seraient recallées en faisant les choses correctement? 

