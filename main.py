"""Telco Customer Churn Prediction

Kaggle-ready notebook source. Each '# %%' section is one notebook cell; copy
the Markdown cells and code cells into a Kaggle Notebook in the same order.
"""

# %% [markdown]
# # Predicting Customer Churn for a Telecom Company
#
# **Business question:** Which customers are likely to cancel their subscription
# next month?  A good predictive model helps the retention team prioritize
# outreach and offers for customers at the highest risk.
#
# This notebook uses a reproducible machine-learning pipeline. It cleans the
# data, splits it into training and test sets, compares two models, evaluates
# the selected model, and saves it for later use.

# %% [markdown]
# ## 1. Import libraries and load the data
#
# In Kaggle, add the Telco Customer Churn dataset to this notebook first. The
# search below finds the CSV regardless of the dataset folder Kaggle assigns.

# %%
from pathlib import Path
import glob
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="deep")
RANDOM_STATE = 42

# Kaggle path first; the local fallback lets this script run from this project.
kaggle_matches = glob.glob("/kaggle/input/**/WA_Fn-UseC_-Telco-Customer-Churn.csv", recursive=True)
local_path = Path("archive/WA_Fn-UseC_-Telco-Customer-Churn.csv")
data_path = Path(kaggle_matches[0]) if kaggle_matches else local_path

df = pd.read_csv(data_path)
print(f"Loaded: {data_path}")
print(f"Dataset shape: {df.shape}")
display(df.head())

# %% [markdown]
# ## 2. Explore data quality and the churn target
#
# We inspect column types, missing values, and the target balance. Churn is a
# binary classification problem: `Yes` means a customer churned and `No` means
# they stayed. The target balance matters because accuracy alone can be
# misleading when one class is more common.

# %%
df.info()
display(df.isna().sum().sort_values(ascending=False).head(10).to_frame("missing_values"))

churn_rate = df["Churn"].value_counts(normalize=True).mul(100).round(2)
print("Churn distribution (%):")
display(churn_rate.to_frame("percentage"))

ax = sns.countplot(data=df, x="Churn")
ax.set_title("Customer Churn Distribution")
ax.bar_label(ax.containers[0])
plt.show()

# %% [markdown]
# ## 3. Clean and prepare the data
#
# `TotalCharges` is stored as text because a few new customers have blank
# values. We convert it to a numeric column; invalid blanks become missing and
# will be imputed inside the pipeline. `customerID` is a unique identifier, so
# it does not describe customer behavior and is removed to prevent memorization.
#
# We turn the target into 1 (churn) and 0 (no churn). A stratified split keeps
# the same approximate churn rate in both the training and unseen test data.

# %%
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

X = df.drop(columns=["Churn", "customerID"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE,
)

numeric_features = X.select_dtypes(include="number").columns.tolist()
categorical_features = X.select_dtypes(exclude="number").columns.tolist()

print(f"Training records: {X_train.shape[0]}")
print(f"Test records: {X_test.shape[0]}")
print(f"Numeric features: {numeric_features}")
print(f"Categorical features: {len(categorical_features)}")

# %% [markdown]
# ## 4. Build reusable preprocessing and model pipelines
#
# Preprocessing is placed inside each pipeline. This is important: the model
# learns imputation, scaling, and category encoding from training data only,
# preventing information leakage from the test set.
#
# We compare:
#
# - **Logistic Regression:** a strong, interpretable baseline for tabular binary classification.
# - **Random Forest:** an ensemble model that can capture non-linear patterns.
#
# ROC-AUC is our selection metric because it measures how well a model ranks
# churners above non-churners across all possible decision thresholds.

# %%
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
}

results = {}
fitted_pipelines = {}

for name, model in models.items():
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)
    test_probabilities = pipeline.predict_proba(X_test)[:, 1]
    results[name] = roc_auc_score(y_test, test_probabilities)
    fitted_pipelines[name] = pipeline

results_df = pd.DataFrame.from_dict(results, orient="index", columns=["Test ROC-AUC"]).sort_values("Test ROC-AUC", ascending=False)
display(results_df.style.format("{:.3f}"))

# %% [markdown]
# ## 5. Evaluate the best model on unseen customers
#
# We select the model with the best test ROC-AUC, then inspect several views of
# performance. The classification report gives precision, recall, and F1-score;
# the confusion matrix shows the raw prediction outcomes. In a retention use
# case, recall for churners is especially useful because it shows how many
# customers at risk the team could identify.

# %%
best_model_name = results_df.index[0]
best_pipeline = fitted_pipelines[best_model_name]
test_probabilities = best_pipeline.predict_proba(X_test)[:, 1]
test_predictions = (test_probabilities >= 0.50).astype(int)

print(f"Selected model: {best_model_name}")
print(f"Test ROC-AUC: {roc_auc_score(y_test, test_probabilities):.3f}")
print("\nClassification report (threshold = 0.50):")
print(classification_report(y_test, test_predictions, target_names=["No Churn", "Churn"]))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ConfusionMatrixDisplay(confusion_matrix(y_test, test_predictions), display_labels=["No Churn", "Churn"]).plot(ax=axes[0], cmap="Blues", colorbar=False)
axes[0].set_title("Confusion Matrix")
RocCurveDisplay.from_predictions(y_test, test_probabilities, ax=axes[1])
axes[1].set_title("ROC Curve")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Explain key churn drivers
#
# Feature importance indicates which inputs most influenced the model's churn
# predictions. For logistic regression, positive coefficients increase churn
# risk and negative coefficients decrease it. For the random forest, values
# show relative predictive importance but not direction.

# %%
feature_names = best_pipeline.named_steps["preprocessor"].get_feature_names_out()
fitted_model = best_pipeline.named_steps["model"]

if best_model_name == "Logistic Regression":
    importance = pd.DataFrame({
        "feature": feature_names,
        "effect_on_churn": fitted_model.coef_[0],
    })
    importance["absolute_effect"] = importance["effect_on_churn"].abs()
    top_features = importance.nlargest(15, "absolute_effect").sort_values("effect_on_churn")
    plot_column = "effect_on_churn"
    plot_title = "Top Logistic Regression Effects on Churn"
else:
    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": fitted_model.feature_importances_,
    })
    top_features = importance.nlargest(15, "importance").sort_values("importance")
    plot_column = "importance"
    plot_title = "Top Random Forest Feature Importances"

plt.figure(figsize=(10, 7))
plt.barh(top_features["feature"], top_features[plot_column], color="#2a9d8f")
plt.title(plot_title)
plt.xlabel(plot_column.replace("_", " ").title())
plt.show()

display(top_features)

# %% [markdown]
# ## 7. Create a prioritized retention list and save the model
#
# The probability threshold can be adjusted to match the budget for retention
# offers. Below, we flag customers with a predicted churn probability of 0.60
# or higher. The final pipeline includes all preprocessing, so it can safely
# score new raw customer records without repeating manual cleaning steps.

# %%
retention_list = X_test.copy()
retention_list["actual_churn"] = y_test.values
retention_list["churn_probability"] = test_probabilities
retention_list["priority_outreach"] = np.where(retention_list["churn_probability"] >= 0.60, "High", "Standard")
retention_list = retention_list.sort_values("churn_probability", ascending=False)

display(retention_list[["churn_probability", "priority_outreach", "actual_churn"]].head(10))
retention_list.to_csv("retention_priority_list.csv", index=False)
joblib.dump(best_pipeline, "telco_churn_model.joblib")

print("Saved: retention_priority_list.csv")
print("Saved: telco_churn_model.joblib")

# %% [markdown]
# ## Business recommendations
#
# 1. Start with the customers at the top of the retention list, where predicted
#    churn probability is highest.
# 2. Tailor offers using the feature analysis—for example, contract type,
#    payment method, service support, and tenure can guide outreach.
# 3. Monitor the model after deployment. Retrain it periodically with new
#    customer behavior and measure whether outreach actually reduces churn.
# 4. Choose the probability threshold with the retention team based on the cost
#    of an offer versus the value of saving a customer.
