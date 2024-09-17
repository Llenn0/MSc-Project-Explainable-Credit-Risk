import time
import dice_ml
import numpy as np
import pandas
from category_encoders import OrdinalEncoder
from matplotlib import pyplot as plt
from dice_custom_models import XGFTModel, FTTModel, TabNetModel, XGBoostModel

"""
    Implements everything required to calculate and visualise Global Counterfactual Importance using DiCE
    counterfactuals.
"""


# Obtain DiCE counterfactuals for a given model, using the entire test set
def dice_counterfactual(model, x_full, y_full, x_test, num_cols, clf_name):
    x_full['class'] = y_full
    d = dice_ml.Data(dataframe=x_full, continuous_features=num_cols, outcome_name='class')

    # Get a custom model depending on what type we are using
    if clf_name == 'FT-Transformer':
        m = FTTModel(model=model, backend='sklearn')
    elif clf_name == 'TabNet':
        m = TabNetModel(model=model, backend='sklearn')
    elif clf_name == 'XGBoost':
        m = XGBoostModel(model=model, backend='sklearn')
    else:
        m = dice_ml.Model(model=model, backend='sklearn')

    # Use the model and data to define explanations
    exp = dice_ml.Dice(data_interface=d, model_interface=m)
    print("Calculating counterfactuals...")

    # Finally generate counterfactuals for the whole test set
    e = exp.generate_counterfactuals(x_test, total_CFs=1, desired_class="opposite", posthoc_sparsity_param=None)
    return e


# Obtain DiCE counterfactuals for the XGFT-Transformer model
def xgft_counterfactual(tab, xg, x_full, y_full, x_test, num_cols):
    x_full['class'] = y_full
    d = dice_ml.Data(dataframe=x_full, continuous_features=num_cols, outcome_name='class')
    m = XGFTModel(tab=tab, xg=xg)
    exp = dice_ml.Dice(data_interface=d, model_interface=m)
    print("Calculating counterfactuals...")
    e = exp.generate_counterfactuals(x_test, total_CFs=1, desired_class="opposite", posthoc_sparsity_param=None)
    return e


# Given a particular counterfactual, calculate normalized distance between it and the original test sample
def get_normalized_distance(counterfactual, cat_cols, k=1.0):
    distance = []
    # For each feature
    for col in counterfactual.test_instance_df:
        # If invalid, skip
        if col == 'class':
            continue
        elif counterfactual.final_cfs_df is None:
            distance.append(0)
            continue
        elif counterfactual.final_cfs_df[col] is None:
            distance.append(0)
            continue

        # Get the maximum and minimum values of the feature
        x_range = counterfactual.data_interface.permitted_range[col]
        x_min = int(x_range[0])
        x_max = int(x_range[-1])
        # Normalize between 0 and 1
        orig = (counterfactual.test_instance_df[col] - x_min) / (x_max - x_min)
        new = (counterfactual.final_cfs_df[col] - x_min) / (x_max - x_min)

        # If categorical, multiply by the categorical scaler provided
        if col in cat_cols:
            distance.append(float(abs(new - orig)) * k)
        else:
            distance.append(float(abs(new - orig)))
    return distance


# Alternative version of the above function for situations where the features need encoded
def get_normalized_distance_raw(counterfactual, cat_cols, encoder, k=1.0):
    distance = []
    # For each feature
    for col in counterfactual.test_instance_df:
        # If invalid, skip
        if col == 'class':
            continue
        elif counterfactual.final_cfs_df is None:
            distance.append(0)
            continue
        elif counterfactual.final_cfs_df[col] is None:
            distance.append(0)
            continue

        # If the feature is categorical, encode it
        if counterfactual.test_instance_df[col].dtype == object:
            counterfactual.test_instance_df = encoder.transform(counterfactual.test_instance_df)
            counterfactual.final_cfs_df = encoder.transform(counterfactual.final_cfs_df)

        # Get the max and min values of the feature
        if counterfactual.test_instance_df[col].dtype == np.int32:
            x_min = int(sorted(counterfactual.data_interface.permitted_range[col])[0][-1])
            x_max = int(sorted(counterfactual.data_interface.permitted_range[col])[-1][-1])
        else:
            x_range = counterfactual.data_interface.permitted_range[col]
            x_min = int(x_range[0])
            x_max = int(x_range[-1])

        # Normalise between 0 and 1
        orig = (counterfactual.test_instance_df[col] - x_min) / (x_max - x_min)
        new = (counterfactual.final_cfs_df[col] - x_min) / (x_max - x_min)

        # If categorical, perform categorical scaling
        if col in cat_cols:
            distance.append(float(abs(new - orig)) * k)
        else:
            distance.append(float(abs(new - orig)))
    return distance


# Main GCI function. Calculates counterfactuals, obtains a matrix of normalized distances, weights them then takes an
# average. Outputs a dataframe containing the average importance of each feature
def global_counterfactuals(model, x_full, y_full, x_test, num_cols, cat_cols, clf_name, k=1.0, max_dist=10.0):
    # XGFT-Transformer requires a slightly different approach
    if clf_name == 'XGFT-Transformer':
        counterfactuals = xgft_counterfactual(model[0], model[1], x_full, y_full, x_test, num_cols)
    else:
        counterfactuals = dice_counterfactual(model, x_full, y_full, x_test, num_cols, clf_name)

    start = time.time()
    dists = np.zeros(x_test.shape)
    # Define the encoder for Transformer models
    encoder = OrdinalEncoder(cols=cat_cols)
    encoder.fit(x_full)

    # For each counterfactual...
    for i in range(0, len(counterfactuals.cf_examples_list)):
        counterfactual = counterfactuals.cf_examples_list[i]
        # Obtain normalized distance for each feature
        if clf_name == 'XGFT-Transformer' or clf_name == 'FT-Transformer':
            dist = get_normalized_distance_raw(counterfactual, cat_cols, encoder, k)
        else:
            dist = get_normalized_distance(counterfactual, cat_cols, k)
        # Weight according to total counterfactual size and add to matrix
        weighted_dist = [i * sum(dist) for i in dist]
        dists[i] = weighted_dist

    # Rescale so that important features have a large value
    dists_df = pandas.DataFrame(data=dists, columns=x_test.columns)
    dists_df = dists_df.applymap(lambda x: abs(x - max_dist) if x > 0 else x)
    # Take the mean and record in dataframe
    average_dists_df = dists_df.mean(axis=0)
    print(f"Calculating global counterfactuals took: {time.time() - start} seconds")
    return average_dists_df


# Plots the dataframe obtained by GCI for clearer viewing and evaluation
def visualise_gci(df):
    df.T.plot(rot=0, kind='barh') # Rotated bar graph
    plt.xlabel("Importance Score")
    plt.title("Global Counterfactual Importance")
    plt.show()
