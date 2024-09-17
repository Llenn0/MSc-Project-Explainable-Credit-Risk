import warnings
import time
import numpy as np
import torch
from imblearn.over_sampling import SMOTE, SMOTENC
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from pytorch_tabular.models import FTTransformerConfig
from pytorch_tabular.config import (
    DataConfig,
    OptimizerConfig,
    TrainerConfig,
)
from pytorch_tabular import TabularModel, TabularModelTuner
import global_counterfactuals
import helpers

print(f"CUDA available: {torch.cuda.is_available()}")
CLF_NAME = "FT-Transformer"  # To identify the model
TEST_NO_CATS = False  # Set to true to get base Transformer performance

# Hyperparameter search space
ftt_search_space = {
    "model_config__input_embed_dim": [8, 16, 32, 64],
    "model_config__num_heads": [4, 6, 8, 10, 12],
    "model_config__num_attn_blocks": [2, 4, 6, 8, 10],
    "model_config__attn_dropout": [0, 0.05, 0.075, 0.1, 0.125, 0.15],
    "model_config__add_norm_dropout": [0, 0.05, 0.075, 0.1, 0.125, 0.15],
    "model_config__ff_dropout": [0, 0.05, 0.075, 0.1, 0.125, 0.15],
    "model_config__learning_rate": [0.0001, 0.0005, 0.001, 0.005],
    "optimizer_config__lr_scheduler": [None, "LinearLR"]
}

# Get user input
DATASET_NAME, USE_SMOTE, OPTIMISE, VALIDATE, CROSS_VAL, EXPLAIN = helpers.user_input()

# If using only base Transformer then encode all to numeric
print("Getting dataset...")
if TEST_NO_CATS:
    x_full, y_full = helpers.import_dataset(DATASET_NAME, USE_SMOTE)
else:
    x_full, y_full = helpers.import_dataset(DATASET_NAME, USE_SMOTE, encode=False)

# Identify names and indices of all cat/num columns
cat_cols, num_cols, cat_idxs, num_idxs = helpers.get_cats(x_full)

if OPTIMISE:

    # Perform SMOTE if necessary
    if USE_SMOTE:
        if TEST_NO_CATS:
            smote = SMOTE(sampling_strategy='minority')
        else:
            smote = SMOTENC(sampling_strategy='minority', categorical_features=cat_idxs)
        x_full, y_full = smote.fit_resample(x_full, y_full)

    # Split data into train and test
    train, test, y_train, y_test = train_test_split(x_full, y_full, test_size=0.2, stratify=y_full)
    train['class'] = y_train
    test['class'] = y_test
    print(f"Dataset Shapes: {train.shape}, {test.shape}")

    # Define config objects for the FT-Transformer
    data_config = DataConfig(
        target=[
            'class'
        ],
        continuous_cols=num_cols,
        categorical_cols=cat_cols,
    )
    trainer_config = TrainerConfig(
        batch_size=1024,
        max_epochs=100,
        early_stopping='valid_loss',
        checkpoints='valid_loss',
        early_stopping_mode='min',
        trainer_kwargs=dict(enable_model_summary=False),
        progress_bar='none'
    )
    optimizer_config = OptimizerConfig()
    model_config = FTTransformerConfig(
        task="classification",
        learning_rate=1e-3,
        metrics=['auroc'],
        metrics_prob_input=[True]
    )

    # Create a tuner object and optimise hyperparameters using the search space and config
    clf = TabularModelTuner(data_config=data_config, trainer_config=trainer_config, optimizer_config=optimizer_config, model_config=model_config)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = clf.tune(train=train, search_space=ftt_search_space, strategy='random_search', n_trials=500, cv=4, metric='auroc', mode='max', progress_bar=True, verbose=True)

    # Record and save results to file
    print("Best Score (AUC): ", result.best_score)
    print("Saving best params to file...")
    if USE_SMOTE:
        filename = f"params/SMOTE/best-params-{DATASET_NAME}-{CLF_NAME}.txt"
    else:
        filename = f"params/BASE/best-params-{DATASET_NAME}-{CLF_NAME}.txt"
    with open(filename, "w") as f:
        params = result.best_params
        for key, value in params.items():
            f.write(f"{key}: {value}\n")


else:
    # Apply SMOTE if necessary
    if USE_SMOTE:
        if TEST_NO_CATS:
            smote = SMOTE(sampling_strategy='minority')
        else:
            smote = SMOTENC(sampling_strategy='minority', categorical_features=cat_idxs)

    # Get the dataset, either with or without validation set
    if CROSS_VAL:
        train, test, y_train, y_test = train_test_split(x_full, y_full, test_size=0.2, stratify=y_full)
        train['class'] = y_train
        test['class'] = y_test
        print(f"Dataset Shapes: {train.shape},{test.shape}")
    else:
        train, test, y_train, y_test = train_test_split(x_full, y_full, test_size=0.25, stratify=y_full)
        val, test, y_val, y_test = train_test_split(test, y_test, test_size=0.5, stratify=y_test)
        train['class'] = y_train
        val['class'] = y_val
        test['class'] = y_test
        print(f"Dataset Shapes: {train.shape}, {val.shape}, {test.shape}")

    # Get best hyperparameters
    if USE_SMOTE:
        best_params_file = f"params/SMOTE/best-params-{DATASET_NAME}-{CLF_NAME}.txt"
    else:
        best_params_file = f"params/BASE/best-params-{DATASET_NAME}-{CLF_NAME}.txt"

    best_params = helpers.get_params_from_file(best_params_file)

    # Define config objects for the FT-Transformer
    data_config = DataConfig(
        target=[
            'class'
        ],
        continuous_cols=num_cols,
        categorical_cols=cat_cols,
    )
    trainer_config = TrainerConfig(
        batch_size=1024,
        max_epochs=100,
        early_stopping='valid_loss',
        early_stopping_mode='min',
        trainer_kwargs=dict(enable_model_summary=False),
        progress_bar='none',
        load_best=False
    )
    optimizer_config = OptimizerConfig(lr_scheduler=best_params['optimizer_config__lr_scheduler'])
    model_config = FTTransformerConfig(
        task="classification",
        learning_rate=best_params['model_config__learning_rate'],
        num_heads=best_params['model_config__num_heads'],
        num_attn_blocks=best_params['model_config__num_attn_blocks'],
        attn_dropout=best_params['model_config__attn_dropout'],
        ff_dropout=best_params['model_config__ff_dropout'],
        add_norm_dropout=best_params['model_config__add_norm_dropout'],
        input_embed_dim=best_params['model_config__input_embed_dim'],
        metrics=['auroc'],
        metrics_prob_input=[True]
    )

    clf = TabularModel(data_config=data_config, trainer_config=trainer_config, optimizer_config=optimizer_config, model_config=model_config)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # To avoid spam
        # For validation, call the appropriate methods
        if VALIDATE:
            if CROSS_VAL:
                cv_scores, oof = clf.cross_validate(cv=4, train=train, metric='auroc', return_oof=True)
                print(f"KFold Mean AUC: {np.mean(cv_scores)}")
            else:
                clf.fit(train=train)
                result = clf.evaluate(val)
                print(result)
        else:
            # For testing, fit to training data then manually obtain test set predictions
            start = time.time()
            clf.fit(train=train)
            train_time = time.time() - start
            start = time.time()
            y_pred = clf.predict(test, ret_logits=True)
            test_time = time.time() - start
            y_prob = y_pred[['1_probability']].values
            y_pred = y_pred[['prediction']].values
            y_test = test[['class']].values

            # Calculate AUC
            auc = roc_auc_score(y_test, y_prob)

            # Calculate the FPR and TPR
            fpr, tpr, thresholds = roc_curve(y_test, y_prob)

            # Calculate the KS statistic
            ks = max(tpr - fpr)

            # Display results and save to file
            print(f"AUC: {auc}, KS: {ks}")
            print("Confusion Matrix: \n\nPredicted label:")
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            print(f"True\t{tn} | {fn}\nlabel:\t{fp} | {tp}")

            if USE_SMOTE:
                smote = "smote"
            else:
                smote = "base"
            with open(f"outputs/scores/{DATASET_NAME}/{smote}/ftt.txt", "a") as f:
                f.write(f"AUC: {auc}, KS: {ks}, Train Time: {train_time}, Test Time: {test_time}\n")
                f.close()

if EXPLAIN:
    print("Explanations...")
    # We don't need the class label for these
    x_train = train.drop('class', axis=1)
    x_test = test.drop('class', axis=1)

    # Obtain SHAP plot
    helpers.shap_plot(clf, x_train, x_test, CLF_NAME)

    # Obtain GCI plot
    start = time.time()
    importances = global_counterfactuals.global_counterfactuals(clf, x_full, y_full, x_test, num_cols, cat_cols, CLF_NAME)
    print(f"GCI took {time.time() - start} seconds.")
    global_counterfactuals.visualise_gci(importances)
