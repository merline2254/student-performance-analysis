import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ML & Preprocessing imports
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from imblearn.over_sampling import SMOTE

# -----------------------------
# Part 1: Exploratory Data Analysis (EDA)
# -----------------------------

# Set seaborn style for better visuals
sns.set_style("whitegrid")

# Load the dataset
df = pd.read_csv("Student skill and Behavior.csv")

# Rename columns for convenience
df.rename(columns={
    "Certification Course": "certification",
    "Gender": "gender",
    "Department": "dep",
    "Height(CM)": "height",
    "Weight(KG)": "weight",
    "10th Mark": "mark10th",
    "12th Mark": "mark12th",
    "college mark": "collegemark",
    "daily studying time": "studytime",
    "prefer to study in": "prefertime",
    "salary expectation": "salexpect",
    "Do you like your degree?": "likedegree",
    "willingness to pursue a career based on their degree": "career_willing",
    "social media & video": "smtime",
    "Travelling Time": "travel",
    "performance Level": "performance",
    "Financial Status": "financial",
    "part-time job": "parttime"
}, inplace=True)

# Convert time-related columns into minutes
def convert_time(x):
    parts = x.split()
    # Extract numeric parts
    times = [int(part) for part in parts if part.isnumeric()]
    if len(times) == 1:
        return times[0] * 60   # assuming it's in hours
    elif len(times) == 2:
        # if text includes "Hour", assume first value in hours and second in minutes
        if "Hour" in x or "hour" in x:
            return times[0] * 60 + times[1]
        else:
            return sum(times)/len(times)
    return np.nan

df["studytime"] = df["studytime"].apply(convert_time)
df["smtime"] = df["smtime"].apply(convert_time)
df["travel"]  = df["travel"].apply(convert_time)

# Remove "%" from 'carrer_willing' and convert to float
df["career_willing"] = df["career_willing"].str.replace("%", "")
df["career_willing"] = df["career_willing"].astype("float")

# -------------------------------------------
# performance-related Visualizations (EDA)
# -------------------------------------------

# Create subplots: 5 graphs in a 3x2 grid (we'll remove the extra subplot)
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 12))
axes = axes.flatten()

# 1. performance Level Distribution
df['performance'].value_counts().plot(kind='bar', ax=axes[0], color="skyblue")
axes[0].set_title("performance Level Distribution")
axes[0].set_xlabel("performance Level")
axes[0].set_ylabel("Count")

# 2. performance vs Gender
df.groupby("performance")["gender"].value_counts().unstack().plot(kind='bar', ax=axes[1], colormap="coolwarm")
axes[1].set_title("performance Level vs Gender")
axes[1].set_xlabel("performance Level")
axes[1].set_ylabel("Count")

# 3. performance vs Department
df.groupby("dep")["performance"].value_counts().unstack().plot(kind='bar', ax=axes[2], colormap="viridis")
axes[2].set_title("performance Level vs Department")
axes[2].set_xlabel("Department")
axes[2].set_ylabel("Count")
axes[2].tick_params(axis='x', rotation=45)

# 4. Financial Status vs performance
df.groupby("financial")["performance"].value_counts().unstack().plot(kind='bar', ax=axes[3], colormap="plasma")
axes[3].set_title("Financial Status vs performance Level")
axes[3].set_xlabel("Financial Status")
axes[3].set_ylabel("Count")

# 5. Part-Time Job vs performance
df.groupby("parttime")["performance"].value_counts().unstack().plot(kind='bar', ax=axes[4], colormap="cool")
axes[4].set_title("Part-Time Job vs performance Level")
axes[4].set_xlabel("Part-Time Job")
axes[4].set_ylabel("Count")

# Remove the extra subplot (6th subplot) if it exists
fig.delaxes(axes[-1])

plt.tight_layout()
plt.show()

# -------------------------------------
# Part 2: Machine Learning (Prediction)
# -------------------------------------

# Map performance levels to ordinal values for modeling
ordinal_mapping = {'Awful': 0, 'Bad': 1, 'Good': 2, 'fabulous': 3}
df['performance'] = df['performance'].map(ordinal_mapping)
#df.drop('performance', axis=1, inplace=True)

# One-hot encode categorical features for modeling
columns_to_encode = ['certification', 'gender', 'dep', 'hobbies', 'prefertime', 'likedegree', 'financial', 'parttime']
encoder = OneHotEncoder(sparse_output=False)
encoded = encoder.fit_transform(df[columns_to_encode])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(columns_to_encode))
df = df.drop(columns=columns_to_encode)
df = pd.concat([df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

# Normalize numerical columns
num_cols = ['height', 'weight', 'mark10th', 'mark12th', 'collegemark', 'studytime', 'salexpect', 'career_willing', 'smtime', 'travel']
scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# Separate features and target variable
X = df.drop('performance', axis=1)
y = df['performance']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle class imbalance using SMOTE on the training data
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Define a RandomForestClassifier and use GridSearchCV to find best hyperparameters
rfc = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
grid_search = GridSearchCV(rfc, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_res, y_train_res)
best_model = grid_search.best_estimator_

print("Best Parameters:", grid_search.best_params_)
print("Best Cross-validation Score:", grid_search.best_score_)

# Make predictions on the test set
y_pred = best_model.predict(X_test)

# Evaluate the predictions
acc = accuracy_score(y_test, y_pred)
print("Test Accuracy:", acc)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Plot Actual vs Predicted performance Levels
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.7, color='teal')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel("Actual performance Level")
plt.ylabel("Predicted performance Level")
plt.title("Actual vs Predicted performance Level")
plt.show()
