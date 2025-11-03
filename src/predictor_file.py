import os
import numpy as np
import hopsworks
import joblib
import logging
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Predict(object):

    def __init__(self, async_logger, model):
        """ Initializes the serving state, reads a trained model"""        
        # Get feature store handle
        project = hopsworks.login()
        self.mr = project.get_model_registry()

        # Retrieve the feature view from the model
        retrieved_model = self.mr.get_model(
            name="xgboost",
            version=1,
        )

        # Load the trained model
        self.hopsworks_model = model

        self.model = joblib.load(os.environ["MODEL_FILES_PATH"] + "/model.joblib")

        logger.info("✅ Model initialization Complete")

    def predict(self, data):
        """ Serves a prediction request usign a trained model"""
        prediction = self.model.predict(data).tolist() # Numpy Arrays are not JSON serializable

        return prediction