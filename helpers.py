import numpy as np
import torch.optim.lr_scheduler
from category_encoders import OrdinalEncoder
from imblearn.over_sampling import SMOTE
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
import pandas as pd
import shap
import matplotlib
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import lime
import lime.lime_tabular
from sklearn.model_selection import StratifiedKFold
import time
from bayes_opt import BayesianOptimization


# Function to read the datasets from a csv file using the parameters to guide filepath
def import_dataset(name, smote, encode=True):
    df = pd.read_csv(f"datasets/{name}/{name}.data.csv")
    X = df.drop('class', axis=1)
    y = df[['class']]
    cat_col_names, num_col_names, cat_idxs, num_idxs = get_cats(X)

    # Only encode with ordinal encoder if encode is True
    if encode:
        print("Encoding categorical data...")
        encoder = OrdinalEncoder(cols=cat_col_names)
        X = encoder.fit_transform(X, y)

        # Oversample if requested
        if smote:
            print("Resampling with SMOTE...")
            smote = SMOTE(sampling_strategy='minority')
            X, y = smote.fit_resample(X, y)

    return X, y


# Function to split a dataset into train, val and test sets
def split_dataset(X, y):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, stratify=y_test)
    y_train = y_train.values.ravel()
    y_val = y_val.values.ravel()
    y_test = y_test.values.ravel()
    print(
        f"Dataset Shapes: {x_train.shape}, {y_train.shape}, {x_val.shape}, {y_val.shape}, {x_test.shape}, {y_test.shape}")
    return x_train, y_train, x_val, y_val, x_test, y_test


# Function to split a dataset into train and test sets
def split_dataset_cv(X, y):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()
    print(f"Dataset Shapes: {x_train.shape}, {y_train.shape}, {x_test.shape}, {y_test.shape}")
    return x_train, y_train, x_test, y_test


# Reads the saved hyperparameter settings from string and transforms into intended formats
def get_params_from_file(filepath):
    params = {}
    with open(filepath) as f:
        for line in f:
            if line == "": break
            (k, v) = line.split()
            k = k.strip(':')
            # Decimal point or e means a float variable, all numbers means an int and other special values
            if '.' in v or 'e-' in v:
                v = float(v)
            elif all(char.isnumeric() for char in v):
                v = int(v)
            elif v == 'None':
                v = None
            elif v == 'True':
                v = True
            elif v == 'False':
                v = False
            params[k] = v
    return params


# Given a dataset, obtain the names and indices of all numeric and categorical columns in a dataframe
def get_cats(x_full):
    cat_col_names = []
    num_col_names = []
    cat_idxs = []
    num_idxs = []

    for col in x_full:
        # If the dtype is object, the column is a categorical feature
        if x_full[col].dtype == 'object':
            cat_col_names.append(col)
            cat_idxs.append(x_full.columns.get_loc(col))
        else:
            num_col_names.append(col)
            num_idxs.append(x_full.columns.get_loc(col))

    return cat_col_names, num_col_names, cat_idxs, num_idxs


# Handle user input to decide which settings to enable - guides all training programs
def user_input():
    correct_input = False
    while not correct_input:
        mode_input = input("Choose dataset to train on:\n1) German\n2) Taiwan\n")
        if mode_input == "1":
            dataset = "german"
            correct_input = True
        elif mode_input == "2":
            dataset = "taiwan"
            correct_input = True
        else:
            print("Incorrect input")

    correct_input = False
    while not correct_input:
        mode_input = input("Use Oversampling?:\n1) Yes\n2) No\n")
        if mode_input == "1":
            smote = True
            correct_input = True
        elif mode_input == "2":
            smote = False
            correct_input = True
        else:
            print("Incorrect input")

    correct_input = False
    while not correct_input:
        mode_input = input("Optimise hyperparameters?:\n1) Yes\n2) No (Use stored best params)\n")
        if mode_input == "1":
            return dataset, smote, True, False, True, False
        elif mode_input == "2":
            opt = False
            correct_input = True
        else:
            print("Incorrect input")

    correct_input = False
    while not correct_input:
        mode_input = input("Validation or test?:\n1) Validation\n2) Test\n")
        if mode_input == "1":
            val = True
            correct_input = True
        elif mode_input == "2":
            val = False
            correct_input = True
        else:
            print("Incorrect input")

    if val:
        correct_input = False
        while not correct_input:
            mode_input = input("Cross Validate or use Validation Set?:\n1) Cross Validate\n2) Validation Set\n")
            if mode_input == "1":
                return dataset, smote, opt, val, True, False
            elif mode_input == "2":
                return dataset, smote, opt, val, False, False
            else:
                print("Incorrect input")
    else:
        correct_input = False
        while not correct_input:
            mode_input = input("Compute Explanations? (Large datasets may take a long time to compute):\n1) Yes\n2) No\n")
            if mode_input == "1":
                return dataset, smote, opt, val, True, True
            elif mode_input == "2":
                return dataset, smote, opt, val, True, False
            else:
                print("Incorrect input")


# Outputs a SHAP violin summary plot on the test set of a given classifier
def shap_plot(clf, x_train, x_test, clf_name):
    start = time.time()

    # Special helper function for XGFTT prediction - encodes the data using tabular model then feeds to XGBoost
    # for probabilities
    def xgft_predict(x_in, tab, xg, col_names):
        x_df = pd.DataFrame(x_in, columns=col_names)
        x_emb = tab.predict(x_df)
        x_emb = x_emb.numpy(force=True)
        return xg.predict_proba(x_emb)

    # Each classifier requires a particular setup. All use KernelSHAP, except for XGBoost which uses TreeSHAP
    if clf_name == 'XGBoost':
        explainer = shap.TreeExplainer(clf)
    elif clf_name == 'TabNet':
        explainer = shap.KernelExplainer(clf.predict_proba,
                                         x_train[np.random.choice(x_train.shape[0], 25, replace=False), :])
    elif clf_name == 'FT-Transformer':
        explainer = shap.KernelExplainer(clf.predict, x_train.sample(n=25), keep_index=True)
    elif clf_name == 'XGFT-Transformer':
        explainer = shap.KernelExplainer(lambda x: xgft_predict(x, tab=clf[0], xg=clf[1], col_names=x_train.columns),
                                         x_train.sample(n=25))
    else:
        explainer = shap.KernelExplainer(clf.predict_proba, x_train.sample(n=25))

    # Obtain Shapley Values
    shap_values = explainer(x_test)
    print(f"SHAP took {time.time() - start} seconds.")

    # Visualise importance in violin plot
    # XGBoost returns margin of trees - equivalent to probability of positive class
    if len(shap_values.shape) > 2:
        shap.plots.violin(shap_values[:, :, 1])
    else:
        shap.plots.violin(shap_values)


# Obtain a lime plot for a given test instance - local explanation only, not used in final report
def lime_plot(clf, x_train, x_test, instance):
    explainer = lime.lime_tabular.LimeTabularExplainer(x_train.values, feature_names=x_train.columns.values.tolist())
    explanation = explainer.explain_instance(x_test.values[instance], clf.predict_proba)
    explanation.save_to_file("outputs/lime-plot.html")


# Unified evaluation function responsible for outputting the desired metrics for a given model and test set
def eval_clf(clf, x_val, y_val, save=False, filepath=None, train_time=None, test_add=None):
    start = time.time()
    # Obtain predictions from the model
    y_pred = clf.predict(x_val)
    y_prob = clf.predict_proba(x_val)[:, 1]
    test_time = time.time() - start

    # Calculate AUC score
    auc = roc_auc_score(y_val, y_prob)

    # Calculate the FPR and TPR
    fpr, tpr, thresholds = roc_curve(y_val, y_prob)
    # Calculate the KS statistic
    ks = max(tpr - fpr)

    # Display results and save to file
    print(f"AUC: {auc}, KS: {ks}")
    print("Confusion Matrix: \n\nTrue label:")
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
    print(f"Pred\t{tn} | {fn}\nlabel:\t{fp} | {tp}")

    if save:
        with open(filepath, "a") as f:
            if test_add != None:
                test_time += test_add
            f.write(f"AUC: {auc}, KS: {ks}, Train Time: {train_time}, Test Time: {test_time}\n")
            f.close()


# Get the cross-validation score for a model, AUC only
def eval_clf_cv(clf, x_train, y_train):
    skf = StratifiedKFold(n_splits=4)
    score = cross_val_score(clf, x_train, y_train, cv=skf, scoring='roc_auc')
    print(f'AUC for each fold: {score}')
    print(f'Average AUC: {"{:.2f}".format(score.mean())}')
    return score.mean()


# Simple unified function for training the given model with the provided training parameters
def train_classifier(clf, train_params):
    print("Training...")
    start = time.time()
    clf.fit(**train_params)
    train_time = time.time() - start
    return clf, train_time


# Optimisation function using random search to obtain ideal hyperparameters
def train_classifier_optimise(x_train, y_train, clf, search_space, dataset, clf_name, smote):
    print("Training and Optimising...")
    opt = RandomizedSearchCV(clf, search_space, cv=StratifiedKFold(n_splits=4), scoring='roc_auc', n_jobs=-1,
                             n_iter=500, verbose=1)

    # Run random search for 500 iterations
    start = time.time()
    opt.fit(x_train, y_train)
    end = time.time()

    # Report results and save to file
    print("Best Estimator: ", opt.best_estimator_)
    print("Best Score (AUC): ", opt.best_score_)

    print("Saving best params to file...")
    if smote:
        smote = "SMOTE"
    else:
        smote = "BASE"
    filename = f"params/{smote}/best-params-{dataset}-{clf_name}.txt"
    with open(filename, "w") as f:
        if clf_name == "XGBoost" or clf_name == "XGFT-Transformer":
            params = opt.best_estimator_.get_xgb_params()
            params["n_estimators"] = opt.best_estimator_.n_estimators
        else:
            params = opt.best_estimator_.get_params()
        for key, value in params.items():
            f.write(f"{key}: {value}\n")

    print(f"Took {(end - start) / 60} minutes.")

    return opt.best_estimator_


# Special optimiser function for TabNet, since it is not compatible with the library used above
def train_tabnet_optimise(x, y, cat_dims, cat_idxs, search_space, dataset, clf_name, smote):
    print("Training and Optimising...")

    # This function describes training TabNet as a single function, which is then provided to the optimiser
    # Given hyperparameters from the optimiser, it trains the model using those settings and returns AUC score
    def tabnet_clf(n_da, n_steps, gamma, n_independent, n_shared, lr, scheduler_gamma, step_size, weight_decay,
                   lambda_sparse):
        # Obtain the correct forms for hyperparameters
        n_d = int(n_da)
        n_a = int(n_da)
        n_steps = int(n_steps)
        n_independent = int(n_independent)
        n_shared = int(n_shared)
        lambda_sparse = float(lambda_sparse)

        skf = StratifiedKFold(n_splits=4)
        aucs = []

        # Perform k-fold cross-validation
        for i, (train_index, test_index) in enumerate(skf.split(x, y)):
            x_train, x_val = x[train_index], x[test_index]
            y_train, y_val = y[train_index], y[test_index]
            # Train the model
            clf = TabNetClassifier(cat_dims=cat_dims, cat_idxs=cat_idxs, n_d=n_d, n_a=n_a, n_steps=n_steps, gamma=gamma,
                                   n_independent=n_independent, n_shared=n_shared, verbose=0,
                                   scheduler_fn=torch.optim.lr_scheduler.StepLR, lambda_sparse=lambda_sparse,
                                   scheduler_params={"gamma": scheduler_gamma, "step_size": step_size},
                                   optimizer_params={"lr": lr, "weight_decay": weight_decay}, device_name='cuda')
            clf.fit(x_train, y_train, eval_metric=['auc'], patience=20)
            # Get probabilities and record AUC scores
            y_pred = clf.predict_proba(x_val)
            val_auc = roc_auc_score(y_score=y_pred[:, 1], y_true=y_val)
            aucs.append(val_auc)

        return np.average(aucs)

    opt = BayesianOptimization(tabnet_clf, search_space)

    # Perform optimisation
    start = time.time()
    opt.maximize(10, 500)
    end = time.time()

    # Display results and save to file
    print("Best Estimator: ", opt.max["params"])
    print("Best Score (AUC): ", opt.max["target"])

    if smote:
        smote = "SMOTE"
    else:
        smote = "BASE"

    print("Saving best params to file...")
    filename = f"params/{smote}/best-params-{dataset}-{clf_name}.txt"
    with open(filename, "w") as f:
        params = opt.max["params"]
        for key, value in params.items():
            # Turn floats into ints where required
            if key == "step_size" or key.startswith("n_"):
                value = int(value)
            f.write(f"{key}: {value}\n")

    print(f"Took {(end - start) / 60} minutes.")
