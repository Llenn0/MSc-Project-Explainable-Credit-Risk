import time
import torch
from sklearn.linear_model import LogisticRegression
import helpers
import global_counterfactuals

print(f"CUDA available: {torch.cuda.is_available()}")
CLF_NAME = "LogRegression"  # To identify the model

# Hyperparameters to search
lr_search_space = [{
    "penalty": [None, 'l1', 'l2'],
    "C": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
    "solver": ['liblinear', 'lbfgs', 'sag'],
    "max_iter": [100, 500, 1000, 1500, 2000, 2500]
}]

# Get user inputs
DATASET_NAME, USE_SMOTE, OPTIMISE, VALIDATE, CROSS_VAL, EXPLAIN = helpers.user_input()
# Import dataset
x_full, y_full = helpers.import_dataset(DATASET_NAME, USE_SMOTE)

# Uncomment to drop a feature, used in xAI performance comparisons
# x_full = x_full.drop(columns='BILL_AMT_AUG')

if OPTIMISE:
    print("Getting dataset...")
    # Split data and perform random search
    x_train, y_train, x_test, y_test = helpers.split_dataset_cv(x_full, y_full)

    clf = helpers.train_classifier_optimise(x_train, y_train, LogisticRegression(), lr_search_space, DATASET_NAME,
                                            CLF_NAME, USE_SMOTE)
else:
    print("Getting dataset...")

    # Get the dataset, either with or without validation set
    if CROSS_VAL:
        x_train, y_train, x_test, y_test = helpers.split_dataset_cv(x_full, y_full)
    else:
        x_train, y_train, x_val, y_val, x_test, y_test = helpers.split_dataset(x_full, y_full)

    # Get the best saved hyperparameters
    if USE_SMOTE:
        best_params_file = f"params/SMOTE/best-params-{DATASET_NAME}-{CLF_NAME}.txt"
    else:
        best_params_file = f"params/BASE/best-params-{DATASET_NAME}-{CLF_NAME}.txt"

    best_params = helpers.get_params_from_file(best_params_file)

    train_params = {
        "X": x_train,
        "y": y_train
    }

    # Train the model
    clf, train_time = helpers.train_classifier(LogisticRegression(**best_params), train_params)

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
        helpers.eval_clf(clf, x_test, y_test, save=True, filepath=f"outputs/scores/{DATASET_NAME}/{smote}/lr.txt",
                         train_time=train_time)

if EXPLAIN:
    print("Explanations...")
    # Produce SHAP plot
    helpers.shap_plot(clf, x_train, x_test, CLF_NAME)

    # Produce GCI plot
    x_full_raw, y_full_raw = helpers.import_dataset(DATASET_NAME, USE_SMOTE, encode=False)
    cat_cols, num_cols, cat_idxs, num_idxs = helpers.get_cats(x_full_raw)
    start = time.time()
    importances = global_counterfactuals.global_counterfactuals(clf, x_full, y_full, x_test, num_cols, cat_cols, CLF_NAME)
    print(f"GCI took {time.time() - start} seconds.")
    global_counterfactuals.visualise_gci(importances)

