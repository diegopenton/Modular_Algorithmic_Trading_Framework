# src/interfaces.py
from typing import Protocol
import pandas as pd

class SignalBranch(Protocol):
    """Contract that all branch models must follow."""

    name: str

    def fit(self, df: pd.DataFrame) -> None:
        """Train the model on a DataFrame of features + labels."""
        ...

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return signals and confidence for the given DataFrame."""
        ...
