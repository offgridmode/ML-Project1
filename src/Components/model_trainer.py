import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_models


# Creating config file for path allocation
@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join(
        "artifacts",
        "model.pkl"
    )


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info(
                "Splitting training and test input data"
            )

            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoost Regressor": CatBoostRegressor(
                    verbose=False
                ),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            model_report: dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models
            )

            logging.info(
                f"Model Report : {model_report}"
            )

            # Get best model score
            best_model_score = max(
                model_report.values()
            )

            # Get best model name
            best_model_name = list(
                model_report.keys()
            )[
                list(model_report.values()).index(
                    best_model_score
                )
            ]

            best_model = models[
                best_model_name
            ]

            if best_model_score < 0.6:
                raise CustomException(
                    "No Best Model Found",
                    sys
                )

            logging.info(
                f"Best Model Found: {best_model_name}"
            )

            # Train best model on full training data
            best_model.fit(
                X_train,
                y_train
            )

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Prediction
            predicted = best_model.predict(
                X_test
            )

            r2_square = r2_score(
                y_test,
                predicted
            )

            logging.info(
                f"R2 Score: {r2_square}"
            )

            return r2_square

        except Exception as e:
            raise CustomException(
                e,
                sys
            )