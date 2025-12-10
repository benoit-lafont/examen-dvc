from typing import NoReturn
import pandas as pd
from sklearn.preprocessing import StandardScaler


def main() -> NoReturn:
    X_train = pd.read_csv("data/processed_data/X_train.csv")
    X_test = pd.read_csv("data/processed_data/X_test.csv")

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.fit_transform(X_test), columns=X_test.columns)
    for file, filename in zip([X_train_scaled, X_test_scaled], ['X_train_scaled', 'X_test_scaled']):
        file.to_csv(f'data/processed_data/{filename}.csv', index=False)

if __name__ == '__main__':
    main()
