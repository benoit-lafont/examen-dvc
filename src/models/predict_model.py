from typing import NoReturn
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error
import json

def main() -> NoReturn:
    X_test = pd.read_csv("data/processed_data/X_test_scaled.csv")
    y_test = pd.read_csv("data/processed_data/y_test.csv")
    loaded_model = joblib.load("models/trained_model.joblib")
    prediction = loaded_model.predict(X_test)
    mse = mean_squared_error(y_test, prediction)
    metrics = {"mean_squared_error": mse}
    prediction_df = pd.DataFrame(prediction, columns=["silica_concentrate"])
    prediction_df.to_csv("data/processed_data/predictions.csv")
    with open('metrics/scores.json', 'w') as f:
        json.dump(metrics, f)

if __name__ == '__main__':
    main()
