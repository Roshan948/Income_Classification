## Income Classification Project – Summary

**The dataset and problem**
For this project I used the Adult Census Income dataset from Kaggle (~48,000 rows), which has data from the 1994 US Census — things like age, education, occupation, marital status, hours worked, etc. The goal is to predict whether someone makes more than $50K a year or not. It's a binary classification problem, and one thing that makes it tricky is that the classes aren't balanced — around 76% of people are in the ≤$50K group, so a model could look "accurate" just by guessing that for everyone without actually learning anything useful.

**What I did**
I started with EDA to understand the data — checking for missing values, looking at how income splits across the different categories, and getting a feel for which features seemed to matter. From there I built out preprocessing, trained a handful of different models to compare, picked the best one, and saved it so it could be used outside the notebook. I also built a small GUI so you can type in someone's info and get a prediction without touching any code.

**Why I made the choices I did**
Since the classes are imbalanced, I used F1 score instead of accuracy to compare models — accuracy would've been misleading here.

For education, instead of just one-hot encoding all 16 raw categories, I grouped them into 6 ordered levels (like No-HS, HS-Grad, up to Grad-Degree). It made more sense to me to encode this as ordinal rather than treating each level as unrelated, since more education is genuinely "more" in a meaningful direction.

I also noticed capital-gain and capital-loss were basically zero for most people but had a few huge outliers, which caused problems for the scaler and made logistic regression struggle to converge. I fixed this by log-transforming those two columns before scaling, which pulled the outliers in without messing up the zeros.

For the actual pipeline, I bundled all the preprocessing steps (encoding + scaling) together with the model into one single scikit-learn Pipeline object, instead of saving separate files for the encoder/scaler/model. This way there's no chance of the preprocessing being slightly different between training and when I actually use the model later.

**How I built it**
Everything was done in Python using scikit-learn for preprocessing and modeling (GridSearchCV with cross-validation), plus PyTorch and Optuna to try out a neural net as an extra comparison. Pandas/matplotlib/seaborn for the EDA and visualizations, joblib to save the trained pipeline, and Tkinter for a simple desktop GUI. I trained everything on Google Colab and then ran the saved model locally through the GUI.

**Results**
The tree-based models (Random Forest and Histogram Gradient Boosting) came out on top compared to logistic regression and k-NN — which makes sense since this dataset has a lot of categorical features and non-linear relationships that tree models handle naturally. The neural net I tried did reasonably well too, but not enough better to justify how much more tuning/compute it needed, so I ended up going with the tree-based model as the final choice.
