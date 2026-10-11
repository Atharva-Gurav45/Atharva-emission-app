import warnings
warnings.filterwarnings("ignore")
import pickle
import json

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

CSV_FILE = "Gas_Turbine_Emission_Classification_Dataset.csv"

# ---------- Train quietly ----------
df = pd.read_csv(CSV_FILE).drop_duplicates()
x = df.drop("Emission_Class", axis=1)
y = df["Emission_Class"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=45, stratify=y
)

scaler = StandardScaler()
x_train_s = pd.DataFrame(scaler.fit_transform(x_train), columns=x.columns, index=x_train.index)
x_train_s["Year"] = x_train["Year"]          # Year stays unscaled

model = KNeighborsClassifier(n_neighbors=5, p=1)
model.fit(x_train_s, y_train)

# ---------- Ask user, predict ----------
def predict_user_input():
    values = {}
    for col in x.columns:
        while True:
            try:
                values[col] = float(input(f"Enter {col}: "))
                break
            except ValueError:
                print("  Please enter a number.")

    row = pd.DataFrame([values], columns=x.columns)
    row_s = pd.DataFrame(scaler.transform(row), columns=x.columns)
    row_s["Year"] = row["Year"]

    print("\nPredicted class:", model.predict(row_s)[0], "\n")


if __name__ == "__main__":
    while True:
        predict_user_input()
        if input("Predict again? (y/n): ").strip().lower() != "y":
            break

# Save model and scaler in Pickle
with open("emission_model.pkl", "wb") as f:
    pickle.dump({"model": model, "scaler": scaler, "columns": list(x.columns)}, f)

# Save model parameters in JSON
with open("emission_model.json", "w") as f:
    json.dump({
        "model": "KNeighborsClassifier",
        "n_neighbors": 5,
        "p": 1,
        "features": list(x.columns)
    }, f, indent=4)
