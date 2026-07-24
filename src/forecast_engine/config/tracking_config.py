from dataclasses import dataclass


@dataclass
class TrackingConfig:
    tracking_uri: str
    experiment_name: str
    use_mlflow: bool
