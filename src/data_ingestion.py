import os
import sys
from logger import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from exception import CustomException   
import psycopg2
import numexpr
os.environ["NUMEXPR_MAX_THREADS"] = "32"

# Postgres connection details
conn = {
    'host': 'localhost',
    'database': 'your_database',
    'user': 'your_user',
    'password': 'your_password',
    'port': '5432'
}


# Read data from PostgreSQL database
def read_data_from_postgres(query):
    try:
        conn = psycopg2.connect(**conn)
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        raise CustomException(f"Error reading data from PostgreSQL: {e}", sys) from e
    
@dataclass
class DataIngestionConfig:
    """
    Data Ingestion Configuration class to store the paths for train and test data.
    """
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')

class DataIngestion:
    """
    Data Ingestion class to handle the ingestion of data.
    """
    def __init__(self):
        self.config = DataIngestionConfig()

    def ingest_data(self):
        """
        Ingest data from PostgreSQL database, split it into train and test sets,
        and save them to specified paths.
        """
        try:
            logging.info("Starting data ingestion process...")
            df = pd.read_csv("../train.csv")
            df = df.tail(100000)  # Limit to the last 100,000 rows for testing purposes
            
            # Create the artifacts directory if it doesn't exist
            os.makedirs(os.path.dirname(self.config.train_data_path), exist_ok=True)

            # Split the data into train and test sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            
            # Save the train and test sets
            train_set.to_csv(self.config.train_data_path, index=False)
            test_set.to_csv(self.config.test_data_path, index=False)

            logging.info("Train and test data saved")

            df.to_csv(self.config.train_data_path, index=False)
            df.to_csv(self.config.test_data_path, index=False)
            logging.info(f"Data ingested successfully and saved to {self.config.train_data_path} and {self.config.test_data_path}")
        except Exception as e:
            raise CustomException(f"Error during data ingestion: {e}", sys) from e
        
if __name__ == "__main__":
    data_ingestion = DataIngestion()
    data_ingestion.ingest_data()
    logging.info("Data ingestion completed successfully.")

