from typing import NoReturn
import pandas as pd
from sklearn.model_selection import train_test_split


def main() -> NoReturn:
    df = pd.read_csv("data/raw_data/raw.csv")
    target = df['silica_concentrate']
    feats = df.drop(['silica_concentrate','date'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(feats, target,
                                                        test_size=0.3,
                                                        random_state=25)
    for file, filename in zip([X_train, X_test, y_train, y_test], ['X_train', 'X_test', 'y_train', 'y_test']):
        file.to_csv(f'data/processed_data/{filename}.csv', index=False)

if __name__ == '__main__':
    main()
