import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    BACKGROUND_NUMERIC_COLUMNS,
    BACKGROUND_PREPROCESSOR_PATH,
    BACKGROUND_TARGET,
    CATEGORICAL_COLUMNS,
    MATH_NUMERIC_COLUMNS,
    MATH_PREPROCESSOR_PATH,
    MATH_TARGET,
    SCORE_COLUMNS,
)
from src.exception import CustomException
from src.logger import logging
from src.pipeline.risk import add_average_score
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = MATH_PREPROCESSOR_PATH
    target_column_name: str = MATH_TARGET
    numerical_columns: List[str] = field(default_factory=lambda: list(MATH_NUMERIC_COLUMNS))
    categorical_columns: List[str] = field(default_factory=lambda: list(CATEGORICAL_COLUMNS))
    derive_average_target: bool = False
    drop_score_columns: bool = False


def math_transformation_config() -> DataTransformationConfig:
    return DataTransformationConfig()


def background_transformation_config() -> DataTransformationConfig:
    return DataTransformationConfig(
        preprocessor_obj_file_path=BACKGROUND_PREPROCESSOR_PATH,
        target_column_name=BACKGROUND_TARGET,
        numerical_columns=list(BACKGROUND_NUMERIC_COLUMNS),
        categorical_columns=list(CATEGORICAL_COLUMNS),
        derive_average_target=True,
        drop_score_columns=True,
    )


class DataTransformation:
    def __init__(self, config: Optional[DataTransformationConfig] = None):
        self.data_transformation_config = config or math_transformation_config()

    def get_data_transformer_object(self):
        """Impute and encode. Numeric columns are scaled; one-hot features are not."""
        try:
            transformers = []
            numerical_columns = self.data_transformation_config.numerical_columns
            categorical_columns = self.data_transformation_config.categorical_columns

            if numerical_columns:
                num_pipeline = Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                )
                transformers.append(("num_pipeline", num_pipeline, numerical_columns))

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "one_hot_encoder",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ),
                ]
            )
            transformers.append(("cat_pipeline", cat_pipeline, categorical_columns))

            preprocessor = ColumnTransformer(transformers)
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def _prepare_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        prepared = df.copy()
        if self.data_transformation_config.derive_average_target:
            prepared = add_average_score(prepared)
        if self.data_transformation_config.drop_score_columns:
            drop_cols = [c for c in SCORE_COLUMNS if c in prepared.columns]
            prepared = prepared.drop(columns=drop_cols)
        return prepared

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = self._prepare_frame(pd.read_csv(train_path))
            test_df = self._prepare_frame(pd.read_csv(test_path))

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = self.data_transformation_config.target_column_name
            input_columns = (
                self.data_transformation_config.numerical_columns
                + self.data_transformation_config.categorical_columns
            )

            input_feature_train_df = train_df[input_columns]
            target_feature_train_df = train_df[target_column_name]
            input_feature_test_df = test_df[input_columns]
            target_feature_test_df = test_df[target_column_name]

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info("Saved preprocessing object")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj,
            )

            feature_names = list(preprocessing_obj.get_feature_names_out())
            return train_arr, test_arr, preprocessing_obj, feature_names, test_df

        except Exception as e:
            raise CustomException(e, sys)
