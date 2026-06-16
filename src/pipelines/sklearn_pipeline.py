from sklearn.pipeline import Pipeline

from src.features.pipeline_preprocessor import (DataPreprocessor)

class PipelineFactory:
    def create_pipeline(
        self,
        model
    ):
        return Pipeline([
            ("preprocessor",DataPreprocessor().build_preprocessor()),
            ("classifier",model)])