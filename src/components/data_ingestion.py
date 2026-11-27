import os
import sys

import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.config import (
    RAW_COPY_PATH,
    RAW_DATA_PATH,
    TEST_DATA_PATH,
    TEST_SIZE,
    TRAIN_DATA_PATH,
    RANDOM_STATE,
)
from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    train_data_path: str = TRAIN_DATA_PATH
    test_data_path: str = TEST_DATA_PATH
    raw_data_path: str = RAW_COPY_PATH
    source_path: str = RAW_DATA_PATH


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv(self.ingestion_config.source_path)
            logging.info("Read the dataset as dataframe")

            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True
            )
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(
                df, test_size=TEST_SIZE, random_state=RANDOM_STATE
            )
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    from src.pipeline.train_pipeline import TrainPipeline

    TrainPipeline().run()
