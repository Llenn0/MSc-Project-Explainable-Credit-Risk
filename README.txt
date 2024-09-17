Thanks for taking the time to look at the codebase for my Masters Project - Explainable Transformers for Credit Risk! In order to run the code, please follow the instructions below.

As detailed in the dissertation, both datasets used in this project were obtained from the UCI Machine Learning Repository and processed as described in dissertation Section 6.1.

The code is written for Python 3.10, though any higher version should also be able to run it just fine. After extracting the code and opening the project in your IDE of choice, you should ensure all project requirements are met. The file requirements.txt contains all the major required libraries to run the code, and it should be possible to automatically or manually install these to your environment. Any other requirements should be dependencies of these, and installed alongside them. If you are having difficulties installing the packages or getting the code to run you can look through the code for the culprit package, or feel free to contact me with any technical difficulties!

The project contains several scripts, but the five files that should be run to reproduce the outputs seen in my dissertation all begin with train_[model name]. If you have a GPU and correct CUDA setup, it should use the GPU, otherwise should revert to CPU. The training scripts operate within the terminal and expect user input to define what they will be doing. Settings include the following:

- Which dataset to use
- Whether to perform oversampling with SMOTE
- Whether to run hyperparameter optimisation or test on saved hyperparameters
- Whether to use validation or test set
- If using validation set, whether to use cross-validation or seperate validation set
- Whether to include SHAP/GCI explanations or not

The params included with the project should be able to reproduce the same results, but as the models are not deterministic there will be some variance and the params can also be overwritten if another optimisation session is performed.

There are comments throughout the code explaining how everything works, but I am happy to answer any additional questions!