import joblib
import numpy as np

from sklearn.ensemble import RandomForestRegressor

from src.config import RANDOM_STATE

class RandomForestModel:
    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
    ):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    def _prepare_input(
        self,
        X,
    ):
        if X.ndim == 3:
            return X.reshape(
                X.shape[0],
                -1,
            )
        return X

    def feature_importance(self):
        return  self.model.feature_importances_

    def train(
            self,
            X_train,
            y_train,
    ):
        X_train = self._prepare_input(X_train)

        self.model.fit(
            X_train,
            y_train,
        )

    def predict(
            self,
            X_test,
    ):
        X_test = self._prepare_input(X_test)
        return self.model.predict(
            X_test
        )

    def save(
            self,
            path,
    ):
        joblib.dump(
            self.model,
            path
        )

    def load(
            self,
            path,
    ):
        self.model = joblib.load(
            path
        )
