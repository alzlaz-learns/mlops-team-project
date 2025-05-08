from pathlib import Path

import pandas as pd
from scipy.io import arff


# called by train_model.
def load_arff_data(filepath: str | Path) -> pd.DataFrame:
    """
    Load and decode an ARFF dataset.

    Parameters:
    - filepath: Path to the ARFF file

    Returns:
    - DataFrame with properly decoded columns
    """
    data, _ = arff.loadarff(filepath)
    df = pd.DataFrame(data)

    # Decode byte strings (common in ARFF files)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].map(lambda x: x.decode() if isinstance(x, bytes) else x)

    return df
