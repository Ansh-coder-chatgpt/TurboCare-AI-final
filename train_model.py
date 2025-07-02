import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor

# 📂 Load dataset
df = pd.read_csv("full_car_data.csv")
df = df.dropna()

# 👇 Add Car_Age column using Year
df["Car_Age"] = 2025 - df["Year"]

# 🎯 Feature and target columns
X = df[["Present_Price", "Kms_Driven", "Owner", "Car_Age", "Fuel_Type", "Seller_Type", "Transmission"]]
y = df["Selling_Price"]

# 🔀 Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ⚙️ Preprocessing
numeric_features = ["Present_Price", "Kms_Driven", "Owner", "Car_Age"]
categorical_features = ["Fuel_Type", "Seller_Type", "Transmission"]

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])

# 🧠 Model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42)),
])

# ✅ Train
model.fit(X_train, y_train)
print("✅ Model trained successfully.")

# 💾 Save model
joblib.dump(model, "car_price_model.pkl")
print("💾 Saved as car_price_model.pkl")

# 📊 Feature Importance
reg = model.named_steps["regressor"]
feature_names = (
    numeric_features +
    list(model.named_steps["preprocessor"]
         .transformers_[1][1]
         .get_feature_names_out(categorical_features))
)
importances = reg.feature_importances_
imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
imp_df.to_csv("feature_importance.csv", index=False)
print("📈 Feature importance saved to CSV.")

# 📈 Save Graph
plt.figure(figsize=(8, 5))
sns.barplot(x="importance", y="feature", data=imp_df.sort_values(by="importance", ascending=False))
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
print("🖼️ Saved feature importance graph as PNG.")
