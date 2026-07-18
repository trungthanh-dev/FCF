import joblib
import sys
import os

from xgboost import XGBRegressor
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import RANDOM_STATE
class XGBoostModel:
    def __init__(
        self,
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
    ):
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
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
        return self.model.feature_importances_

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
