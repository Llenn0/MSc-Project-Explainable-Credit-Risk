import time
import warnings
import torch
import xgboost
from imblearn.over_sampling import SMOTENC
from sklearn.model_selection import train_test_split
from pytorch_tabular.models import FTTransformerConfig
from pytorch_tabular.config import (
    DataConfig,
    OptimizerConfig,
    TrainerConfig,
)
import global_counterfactuals
import helpers
from custom_tabular_model import CustomTabularModel

print(f"CUDA available: {torch.cuda.is_available()}")
CLF_NAME = "XGFT-Transformer"  # To identify the model

# Hyperparameter search spaces
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

xgboost_search_space = [{
    "n_estimators": [100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000],
    "max_depth": [2, 3, 4, 5, 6, 7, 8, 9, 10],
    "learning_rate": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
    "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1],
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1],
    "reg_alpha": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    "reg_lambda": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    "gamma": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
}]

# Get user input
DATASET_NAME, USE_SMOTE, OPTIMISE, VALIDATE, CROSS_VAL, EXPLAIN = helpers.user_input()

print("Getting dataset...")
x_full, y_full = helpers.import_dataset(DATASET_NAME, USE_SMOTE, encode=False)

# Get columns/indices of numeric and categorical data
cat_cols, num_cols, cat_idxs, num_idxs = helpers.get_cats(x_full)

# Apply SMOTE if necessary
if USE_SMOTE:
    smote = SMOTENC(sampling_strategy='minority', categorical_features=cat_idxs)
    x_full, y_full = smote.fit_resample(x_full, y_full)

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
    best_params_file_xg = f"params/SMOTE/best-params-{DATASET_NAME}-XGBoost.txt"
    best_params_file_ftt = f"params/SMOTE/best-params-{DATASET_NAME}-FT-Transformer.txt"
else:
    best_params_file_xg = f"params/BASE/best-params-{DATASET_NAME}-XGBoost.txt"
    best_params_file_ftt = f"params/BASE/best-params-{DATASET_NAME}-FT-Transformer.txt"

best_params_xg = helpers.get_params_from_file(best_params_file_xg)
best_params_ftt = helpers.get_params_from_file(best_params_file_ftt)

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
optimizer_config = OptimizerConfig(lr_scheduler=best_params_ftt['optimizer_config__lr_scheduler'])
model_config = FTTransformerConfig(
    task="classification",
    learning_rate=best_params_ftt['model_config__learning_rate'],
    num_heads=best_params_ftt['model_config__num_heads'],
    num_attn_blocks=best_params_ftt['model_config__num_attn_blocks'],
    attn_dropout=best_params_ftt['model_config__attn_dropout'],
    ff_dropout=best_params_ftt['model_config__ff_dropout'],
    add_norm_dropout=best_params_ftt['model_config__add_norm_dropout'],
    input_embed_dim=best_params_ftt['model_config__input_embed_dim'],
    metrics=['auroc'],
    metrics_prob_input=[True],
    head='LinearHead',
    head_config={'layers': '64', 'activation': 'ReLU', 'dropout': 0.0, 'use_batch_norm': False,
                 'initialization': 'kaiming'}
)

# Train the FTTransformer and get the training embeddings for XGBoost
tab = CustomTabularModel(data_config=data_config, trainer_config=trainer_config, optimizer_config=optimizer_config,
                         model_config=model_config)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    start = time.time()
    tab.fit(train=train)
    tab.behead_model()
    train_embeddings = tab.predict(train)
    train_embeddings.numpy(force=True)
    train_time = time.time() - start
    y_train = train[['class']].values

# For hyperparameter optimisation, optimise the XGBoost parameters using standard function
if OPTIMISE:
    clf = helpers.train_classifier_optimise(train_embeddings, y_train, xgboost.XGBClassifier(eval_metric='auc'),
                                            xgboost_search_space, DATASET_NAME, CLF_NAME, USE_SMOTE)
elif VALIDATE:
    # For evaluation, train using best hyperparameters and embeddings from the FTT
    train_params = {
        "X": train_embeddings,
        "y": y_train
    }
    xg = helpers.train_classifier(xgboost.XGBClassifier(**best_params_xg), train_params)

    if CROSS_VAL:
        helpers.eval_clf_cv(xg, train_embeddings, y_train)
    else:
        # Get validation embeddings from tabular model and use for evaluation
        val_embeddings = tab.predict(val)
        val_embeddings.numpy(force=True)
        y_val = val[['class']].values
        helpers.eval_clf(xg, val_embeddings, y_val)
else:
    train_params = {
        "X": train_embeddings,
        "y": y_train
    }
    xg, train_time_xg = helpers.train_classifier(xgboost.XGBClassifier(**best_params_xg), train_params)
    train_time += train_time_xg

    # Get test embeddings from tabular model and use for evaluation
    start = time.time()
    test_embeddings = tab.predict(test)
    test_embeddings = test_embeddings.numpy(force=True)
    test_time = time.time() - start
    y_test = test[['class']].values
    if USE_SMOTE:
        smote = "smote"
    else:
        smote = "base"
    helpers.eval_clf(xg, test_embeddings, y_test, save=True, filepath=f"outputs/scores/{DATASET_NAME}/{smote}/xgft.txt", train_time=train_time, test_add=test_time)

    if EXPLAIN:
        print("Explanations...")
        x_train = train.drop('class', axis=1)
        x_test = test.drop('class', axis=1)

        # Obtain SHAP plot
        helpers.shap_plot((tab, xg), x_train, x_test, CLF_NAME)

        # Obtain GCI plot
        start = time.time()
        importances = global_counterfactuals.global_counterfactuals((tab, xg), x_full, y_full, x_test, num_cols, cat_cols, CLF_NAME)
        print(f"GCI took {time.time() - start} seconds.")
        global_counterfactuals.visualise_gci(importances)
