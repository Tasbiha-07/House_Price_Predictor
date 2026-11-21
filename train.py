import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error, r2_score

# 1. Load dataset
df = pd.read_csv("Real_estate - Real_estate.csv")
if "No" in df.columns:
    df = df.drop(columns=["No"])

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Define candidate models
models = {
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=42)
}

results = []
best_pipeline = None
best_r2 = -float("inf")
best_model_name = ""

# 4. Train and evaluate all models
for name, regressor in models.items():
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", regressor)
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    
    rmse = root_mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    results.append({"Model": name, "R² Score": round(r2, 4), "RMSE": round(rmse, 2)})
    
    if r2 > best_r2:
        best_r2 = r2
        best_pipeline = pipe
        best_model_name = name

results_df = pd.DataFrame(results).sort_values(by="R² Score", ascending=False)
print("\nModel Comparison:")
print(results_df.to_string(index=False))

# 5. Extract feature importances from the winning tree model
importances = best_pipeline.named_steps["model"].feature_importances_
feat_imp_df = pd.DataFrame({
    "Feature": ["Transaction Date", "House Age", "MRT Distance", "Convenience Stores", "Latitude", "Longitude"],
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

# 6. Save everything in one artifact
joblib.dump({
    "pipeline": best_pipeline,
    "best_model_name": best_model_name,
    "columns": X.columns.tolist(),
    "metrics": results_df,
    "feature_importance": feat_imp_df
}, "house_price_model.pkl")

print(f"\nSaved winning model ({best_model_name}) to house_price_model.pkl")