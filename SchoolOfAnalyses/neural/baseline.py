import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


def main():
    train = pd.read_csv('train.csv')
    X_train = train.drop('y', axis=1)
    y_train = train['y'].values

    X_test = pd.read_csv('test_x.csv')

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestRegressor(
        verbose=False
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    pd.Series(predictions, name='y').to_csv('test_y.csv', index=False)


if __name__ == '__main__':
    main()