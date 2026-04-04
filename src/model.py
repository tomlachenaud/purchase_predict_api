import os
import joblib
import mlflow
from mlflow.tracking import MlflowClient

ENV = os.getenv("ENV")
MLFLOW_REGISTRY_NAME = os.getenv("MLFLOW_REGISTRY_NAME")

# Le warning est déjà guardé par le fichier __init__.py, pas besoin de le répéter ici
mlflow.set_tracking_uri(os.getenv("MLFLOW_SERVER"))  # ty: ignore[invalid-argument-type]

class Model:
    def __init__(self):
        self.model = None
        self.transform_pipeline = None
        self.load_model()

    def load_model(self):
        # We query currently staging or production model, according to environment specification
        client = MlflowClient()
        alias = ENV
        model_version = client.get_model_version_by_alias(
            os.getenv("MLFLOW_REGISTRY_NAME"),  # ty: ignore[invalid-argument-type]
            alias,  # ty: ignore[invalid-argument-type]
        )

        # In MLFlow v3, construct the artifact URI and use mlflow.artifacts.download_artifacts()
        artifact_uri = f"runs:/{model_version.run_id}/transform_pipeline.pkl"
        pipeline_path = mlflow.artifacts.download_artifacts(
            artifact_uri=artifact_uri
        )  # ty: ignore[possibly-missing-attribute]

        if pipeline_path is None:
            raise RuntimeError(
                f"Failed to download transform_pipeline.pkl for run_id={model_version.run_id}. "
                "The artifact was not found. Ensure the training pipeline logs "
                "transform_pipeline.pkl via mlflow.log_artifact()."
            )

        self.model = mlflow.sklearn.load_model(
            f"models:/{MLFLOW_REGISTRY_NAME}@{alias}"
        )
        # We must also retrieve transform pipeline from artifacts
        self.transform_pipeline = joblib.load(pipeline_path)

    def predict(self, X):
        if self.model:
            if self.transform_pipeline:
                for name, encoder in self.transform_pipeline.items():
                    X[name] = X[name].fillna("unknown")
                    X[name] = encoder.transform(X[name])
            for col in ["user_id", "user_session", "purchased"]:
                if col in X:
                    X = X.drop(col, axis=1)
            return self.model.predict(X)
        return None
