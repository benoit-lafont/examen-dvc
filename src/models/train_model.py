from typing import NoReturn
import pandas as pd
from sklearn.linear_model import Ridge
import pickle
import joblib

def main() -> NoReturn:
    X_train = pd.read_csv("data/processed_data/X_train_scaled.csv")
    y_train = pd.read_csv("data/processed_data/y_train.csv")
    with open("models/best_params.pck", "rb") as f:
        params = pickle.load(f)

    model = Ridge(alpha=params['alpha'])

    model.fit(X_train, y_train)
    model_filename = './models/trained_model.joblib'
    joblib.dump(model, model_filename)

if __name__ == '__main__':
    main()
