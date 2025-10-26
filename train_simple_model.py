import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import joblib
import os
import warnings

warnings.filterwarnings("ignore")
print("Starting SIMPLE model training script...")

# --- 1. Load Data ---
try:
    df = pd.read_csv("/Users/hemaharini/dashboard/autism_data.csv")
    print("Data loaded.")
except FileNotFoundError:
    print("Error: 'dataset/autism_data.csv' not found.")
    exit()

# --- 2. (NEW) Define ALL columns we need ---
# These are the 10 scores, 5 demographic fields, and 1 target
columns_to_keep = [
    "A1_Score",
    "A2_Score",
    "A3_Score",
    "A4_Score",
    "A5_Score",
    "A6_Score",
    "A7_Score",
    "A8_Score",
    "A9_Score",
    "A10_Score",
    "age",
    "gender",
    "jundice",
    "austim",
    "relation",
    "Class/ASD",
]

# Select *only* these columns
try:
    df = df[columns_to_keep]
    print("Kept only necessary columns.")
except KeyError:
    print("Error: Your CSV is missing one of the required columns.")
    print(f"Required: {columns_to_keep}")
    exit()


# --- 3. Feature Engineering & Cleaning ---
df["age"] = pd.to_numeric(df["age"], errors="coerce")

score_cols = [f"A{i}_Score" for i in range(1, 11)]
df["Total_AQ_Score"] = df[score_cols].sum(axis=1)

# --- 4. Handle Missing Values ---
df["age"] = df["age"].fillna(df["age"].median())
for col in df.columns:
    if col != "age":
        df[col] = df[col].fillna(df[col].mode()[0])
print("Missing values handled.")

# --- 5. Encode Categorical Features ---
df["Class/ASD"] = df["Class/ASD"].map({"NO": 0, "YES": 1})

label_encoders = {}
for col in df.select_dtypes(["object", "category"]):
    if col != "Class/ASD":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
print("Features encoded.")

# --- 6. Define Features (X) and Target (y) ---
y = df["Class/ASD"]
X = df.drop("Class/ASD", axis=1)
model_columns = X.columns
print(f"Model will be trained on columns: {list(model_columns)}")

# --- 7. Split Data ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 8. Scale Numerical Data ---
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("Data scaled.")

# --- 9. Train the SIMPLE Model ---
model = DecisionTreeClassifier(max_depth=4, class_weight="balanced", random_state=42)
model.fit(X_train, y_train)
print("Simple Decision Tree model trained.")

# --- 10. Evaluate Model ---
y_pred = model.predict(X_test)
print("\n--- Model Evaluation ---")
print(classification_report(y_test, y_pred, target_names=["NO", "YES"]))

# --- 11. Save Model and Preprocessing Artifacts ---
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model.joblib")
joblib.dump(scaler, "model/scaler.joblib")
joblib.dump(label_encoders, "model/encoders.joblib")
joblib.dump(model_columns, "model/model_columns.joblib")

print("\nAll files saved to 'model/' directory.")
print("Training script finished.")
