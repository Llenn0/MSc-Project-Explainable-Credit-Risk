import time

import torch
from category_encoders import TargetEncoder
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from skopt.space import Real, Integer, Categorical
import helpers
import global_counterfactuals as g_cnt

print(f"CUDA available: {torch.cuda.is_available()}")
CLF_NAME = "Forest"

rf_search_space = {
    "model__n_estimators": Integer(500, 2000),
    "model__min_samples_leaf": Integer(1, 4),
    "model__min_samples_split": Integer(2, 10),
    "model__bootstrap": Categorical([True, False])
}

# Get user inputs to decide what to do
DATASET_NAME, USE_SMOTE, OPTIMISE, VALIDATE, CROSS_VAL = helpers.user_input()

# Bayesian Hyperparameter Optimisation
if OPTIMISE:
    print("Getting dataset...")
    x_train, y_train, x_test, y_test = helpers.split_dataset_cv(DATASET_NAME, USE_SMOTE)

    clf = helpers.train_classifier_optimise(x_train, y_train, RandomForestClassifier(), rf_search_space, DATASET_NAME,
                                            CLF_NAME, USE_SMOTE)
else:
    print("Getting dataset...")

    # Get the dataset, either with or without validation set
    if CROSS_VAL:
        x_train, y_train, x_test, y_test = helpers.split_dataset_cv(DATASET_NAME, USE_SMOTE)
    else:
        x_train, y_train, x_val, y_val, x_test, y_test = helpers.split_dataset(DATASET_NAME, USE_SMOTE)

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

    # Train Random Forest
    clf, train_time = helpers.train_classifier(RandomForestClassifier(**best_params), train_params)

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
        helpers.eval_clf(clf, x_test, y_test, save=True, filepath=f"outputs/scores/{DATASET_NAME}/{smote}/forest.txt", train_time=train_time)

# print("Explanations...")
# helpers.shap_plot(clf, x_test)
# helpers.lime_plot(clf, x_train, x_test, 5)

x_full, y_full = helpers.import_dataset(DATASET_NAME, USE_SMOTE, encode=False)

cat_cols, num_cols, cat_idxs, num_idxs = helpers.get_cats(x_full)

g_cnt.dice_counterfactual(clf, x_train, y_train, x_test, y_test, num_cols, CLF_NAME)