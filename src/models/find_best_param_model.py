from typing import NoReturn
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
import pickle
import os

def main() -> NoReturn:
    X_train = pd.read_csv("data/processed_data/X_train_scaled.csv")
    y_train = pd.read_csv("data/processed_data/y_train.csv")
    param_grid = {
        "alpha": [0.001, 0.01, 0.1, 0.3, 0.7, 1, 10, 50, 100]
    }

    model = GridSearchCV(
        estimator=Ridge(),
        param_grid=param_grid,
        cv=5,
        scoring="neg_mean_squared_error"
    )
    model.fit(X_train, y_train)
    if os.path.exists('models') == False:
        os.makedirs('models')
    with open("models/best_params.pck", "wb") as f:
        pickle.dump(model.best_params_, f)


if __name__ == '__main__':
    main()
