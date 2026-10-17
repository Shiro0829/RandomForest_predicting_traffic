import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score


# Loading the dataset
df = pd.read_csv('UNSW_NB15_balanced_30k.csv')

# Features (X) - Drop id and both answer keys
X = df.drop(columns=['id', 'label', 'attack_cat'])

# Targets (y)
y_bin = df['label']
y_multi = df['attack_cat']


# Identifing categorical columns
categorical_cols = ['proto', 'service', 'state']

# Setting up the OneHotEncoder inside a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough' # Leaves the 39 numeric columns untouched
)


# Binary Pipeline
binaryPipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier
     (n_estimators=20, 
      max_depth=10,
      max_leaf_nodes=300, 
      min_samples_leaf=3,
      random_state=42))
])

# Multiclass Pipeline
multiclassPipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier
     (n_estimators=75, 
      max_depth=None,
      max_leaf_nodes=300,
      min_samples_split=8,
      min_samples_leaf=2,
      max_features=0.5,
      class_weight='balanced',  
      random_state=42))
])


# Splitting the data
X_train, X_val, y_bin_train, y_bin_val, y_multi_train, y_multi_val = train_test_split(
    X, y_bin, y_multi, test_size=0.2, random_state=42, stratify=y_multi
)


print("=======Binary Model Evaluation =====")
binaryPipeline.fit(X_train, y_bin_train)
y_bin_pred = binaryPipeline.predict(X_val)

print(f"Accuracy:{accuracy_score(y_bin_val, y_bin_pred):.4f}")
print(f"Precision:{precision_score(y_bin_val, y_bin_pred):.4f}")
print(f"Recall:   {recall_score(y_bin_val, y_bin_pred):.4f}")
print(f"F1 Score: {f1_score(y_bin_val, y_bin_pred):.4f}\n")


print("======Multiclass Model Evaluation =====")
multiclassPipeline.fit(X_train, y_multi_train)
y_multi_pred = multiclassPipeline.predict(X_val)


print(classification_report(y_multi_val, y_multi_pred, digits=4))

# Refitting on the full dataset (X, y_bin, y_multi)
binaryPipeline.fit(X, y_bin)
multiclassPipeline.fit(X, y_multi)

joblib.dump({
    'binaryModel': binaryPipeline,
    'multiclassModel': multiclassPipeline,
}, 'randomForestModel.joblib')

print("Models saved successfully to randomForestModel.joblib")

# Verification Check
models = joblib.load('randomForestModel.joblib')
raw_test_data = df.drop(columns=['id', 'label', 'attack_cat']).head(5)

print("\n--- Artifact Verification ---")
print("Binary Predictions:", models['binaryModel'].predict(raw_test_data))
print("Multiclass Predictions:", models['multiclassModel'].predict(raw_test_data))