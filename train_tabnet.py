from sklearn.model_selection import train_test_split
import global_counterfactuals
import time
import helpers
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
from imblearn.over_sampling import SMOTE

print(f"CUDA available: {torch.cuda.is_available()}")
CLF_NAME = "TabNet"  # To identify the model

# Performs basic categorical encoding on the dataset and returns the names/indices of categorical features
def categorical_encoding(x):
    types = x.dtypes
    categorical_columns = []
    categorical_dims = {}
    for col in x.columns:
        # Categorical columns will be objects in the dataframe
        if types[col] == 'object':
            print(col, x[col].nunique())
            l_enc = LabelEncoder()
            x[col] = x[col].fillna("VV_likely")
            x[col] = l_enc.fit_transform(x[col].values)
            categorical_columns.append(col)
            categorical_dims[col] = len(l_enc.classes_)

    print(f"Categorical columns: {categorical_columns}")
    print(f"Categorical dims: {categorical_dims}")

    features = [col for col in x.columns]
    cat_idxs = [i for i, f in enumerate(features) if f in categorical_columns]
    cat_dims = [categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]

    return x, cat_idxs, cat_dims


# Get user input
DATASET_NAME, USE_SMOTE, OPTIMISE, VALIDATE, CROSS_VAL, EXPLAIN = helpers.user_input()

if OPTIMISE:
    print("Getting dataset...")
    x_full, y_full = helpers.import_dataset(DATASET_NAME, USE_SMOTE, encode=False)

    print("Encoding categorical data...")
    x_full_encoded, cat_idxs, cat_dims = categorical_encoding(x_full)

    # Apply SMOTE if necessary
    if USE_SMOTE:
        smote = SMOTE(sampling_strategy='minority')
        x_full_encoded, y_full = smote.fit_resample(x_full_encoded, y_full)

    # Hyperparameter search space
    tabnet_search_space = {
        "n_da": (2, 64),
        "n_steps": (2, 10),
        "gamma": (1.0, 2.0),
        "n_independent": (1, 5),
        "n_shared": (1, 5),
        "lr": (0.0001, 0.01),
        "scheduler_gamma": (0.9, 0.999),
        "step_size": (1, 15),
        "weight_decay": (0.0001, 0.1),
        "lambda_sparse": (0.00001, 0.1)
    }

    # Split dataset into train and test and turn to numpy arrays
    x_train, x_test, y_train, y_test = train_test_split(x_full_encoded, y_full, test_size=0.2, stratify=y_full)
    x_train = x_train.to_numpy()
    x_test = x_test.to_numpy()
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()
    print(f"Dataset Shapes: {x_train.shape}, {y_train.shape}, {x_test.shape}, {y_test.shape}")
    warnings.filterwarnings("ignore") # To avoid getting spammed

    # Hyperparameter optimisation
    helpers.train_tabnet_optimise(x_train, y_train, cat_dims, cat_idxs, tabnet_search_space, DATASET_NAME, CLF_NAME, USE_SMOTE)

else:
    print("Getting dataset...")
    x_full, y_full = helpers.import_dataset(DATASET_NAME, USE_SMOTE, encode=False)

    print("Encoding categorical data...")
    x_full_encoded, cat_idxs, cat_dims = categorical_encoding(x_full)

    # Perform SMOTE if necessary
    if USE_SMOTE:
        smote = SMOTE(sampling_strategy='minority')
        x_full_encoded, y_full = smote.fit_resample(x_full_encoded, y_full)

    # Get the dataset, either with or without validation set
    if CROSS_VAL:
        x_train, x_test, y_train, y_test = train_test_split(x_full_encoded, y_full, test_size=0.2, stratify=y_full)
        print(f"Dataset Shapes: {x_train.shape}, {y_train.shape}, {x_test.shape}, {y_test.shape}")
    else:
        x_train, x_test, y_train, y_test = train_test_split(x_full_encoded, y_full, test_size=0.25, stratify=y_full)
        x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, stratify=y_test)
        x_val = x_val.to_numpy()
        y_val = y_val.values.ravel()
        print(f"Dataset Shapes: {x_train.shape}, {y_train.shape}, {x_val.shape}, {y_val.shape}, {x_test.shape}, {y_test.shape}")

    x_train = x_train.to_numpy()
    y_train = y_train.values.ravel()

    # Get the best hyperparameters
    if USE_SMOTE:
        best_params_file = f"params/SMOTE/best-params-{DATASET_NAME}-{CLF_NAME}.txt"
    else:
        best_params_file = f"params/BASE/best-params-{DATASET_NAME}-{CLF_NAME}.txt"

    # Some hyperparameters are saved with different names, so unpack them into their required formats
    best_params = helpers.get_params_from_file(best_params_file)
    best_params["optimizer_params"] = {"lr": best_params["lr"], "weight_decay": best_params["weight_decay"]}
    best_params["scheduler_params"] = {"gamma": best_params["scheduler_gamma"], "step_size": best_params["step_size"]}
    best_params["n_d"] = best_params["n_da"]
    best_params["n_a"] = best_params["n_da"]
    del best_params["lr"]
    del best_params["step_size"]
    del best_params["scheduler_gamma"]
    del best_params["n_da"]
    del best_params["weight_decay"]

    train_params = {
        "X_train": x_train,
        "y_train": y_train,
        "eval_set": [(x_train, y_train)],
        "eval_name": ['train'],
        "eval_metric": ['auc']
    }

    # Train the TabNet classifier
    clf, train_time = helpers.train_classifier(TabNetClassifier(**best_params), train_params)

    # Evaluate using standard metrics
    print("Evaluating...")
    if VALIDATE:
        if CROSS_VAL:
            helpers.eval_clf_cv(clf, x_train, y_train)
        else:
            helpers.eval_clf(clf, x_val, y_val)
    else:
        if USE_SMOTE:
            smote = "smote"
        else:
            smote = "base"
        helpers.eval_clf(clf, x_test.values, y_test.values, save=True, filepath=f"outputs/scores/{DATASET_NAME}/{smote}/tabnet.txt", train_time=train_time)

    if EXPLAIN:
        print("Explanations...")
        # Produce SHAP plot
        helpers.shap_plot(clf, x_train, x_test, CLF_NAME)

        # Produce GCI plot
        cat_cols, num_cols, cat_idxs, num_idxs = helpers.get_cats(x_full)
        start = time.time()
        importances = global_counterfactuals.global_counterfactuals(clf, x_full_encoded, y_full, x_test, num_cols, cat_cols, CLF_NAME)
        print(f"GCI took {time.time() - start} seconds.")
        global_counterfactuals.visualise_gci(importances)
