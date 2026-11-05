"""
src/data/loader.py
──────────────────
CSV ingestion with validation and lightweight sanitization.
Returns a raw DataFrame — irreversible column removal is never performed here.
"""

import io
import logging
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DataLoader:
    """Loads CSV data from a file path or an in-memory bytes buffer."""

    SUPPORTED_ENCODINGS = ["utf-8", "latin-1", "iso-8859-1", "cp1252"]

    def __init__(self) -> None:
        self._raw_df: Optional[pd.DataFrame] = None
        self._source_name: str = ""

    # ── Public API ────────────────────────────────────────────────────────────

    def load_from_path(self, path: Union[str, Path]) -> pd.DataFrame:
        """Load CSV from a file path on disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        if path.suffix.lower() not in {".csv", ".tsv"}:
            raise ValueError(f"Unsupported file type: {path.suffix}. Only .csv and .tsv are supported.")

        self._source_name = path.name
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        df = self._try_load(path, sep=sep)
        self._raw_df = df
        return df

    def load_from_bytes(self, content: bytes, filename: str = "upload.csv") -> pd.DataFrame:
        """Load CSV from a bytes buffer (e.g., Streamlit uploaded file)."""
        self._source_name = filename
        suffix = Path(filename).suffix.lower()
        sep = "\t" if suffix == ".tsv" else ","
        df = self._try_load(io.BytesIO(content), sep=sep)
        self._raw_df = df
        return df

    def load_from_file_object(self, file_obj, filename: str = "upload.csv") -> pd.DataFrame:
        """Load CSV from a file-like object (supports Streamlit UploadedFile)."""
        self._source_name = filename
        suffix = Path(filename).suffix.lower()
        sep = "\t" if suffix == ".tsv" else ","
        df = self._try_load(file_obj, sep=sep)
        self._raw_df = df
        return df

    @property
    def source_name(self) -> str:
        return self._source_name

    # ── Private helpers ───────────────────────────────────────────────────────

    def _try_load(self, source, sep: str = ",") -> pd.DataFrame:
        """Try multiple encodings when loading CSV."""
        last_error: Optional[Exception] = None

        for encoding in self.SUPPORTED_ENCODINGS:
            try:
                if isinstance(source, (str, Path)):
                    df = pd.read_csv(source, sep=sep, encoding=encoding, low_memory=False)
                else:
                    # Reset buffer position before each attempt
                    if hasattr(source, "seek"):
                        source.seek(0)
                    df = pd.read_csv(source, sep=sep, encoding=encoding, low_memory=False)

                self._validate(df)
                df = self._sanitize(df)
                logger.info(
                    "Loaded %s: %d rows × %d columns (encoding=%s)",
                    self._source_name,
                    df.shape[0],
                    df.shape[1],
                    encoding,
                )
                return df
            except (UnicodeDecodeError, LookupError):
                last_error = None
                continue
            except pd.errors.EmptyDataError as exc:
                # Keep this distinct from malformed CSV: callers can give the
                # user a useful recovery action for an empty upload.
                raise ValueError("The uploaded file is empty.") from exc
            except pd.errors.ParserError as exc:
                raise ValueError(f"Could not parse CSV file: {exc}") from exc

        raise ValueError(
            f"Could not read '{self._source_name}' with any supported encoding "
            f"({', '.join(self.SUPPORTED_ENCODINGS)}). "
            f"Last error: {last_error}"
        )

    @staticmethod
    def _validate(df: pd.DataFrame) -> None:
        if df is None or df.empty:
            raise ValueError("The uploaded file is empty.")
        if df.shape[0] < 2:
            raise ValueError("Dataset must have at least 2 rows.")
        if df.shape[1] < 2:
            raise ValueError("Dataset must have at least 2 columns.")

    @staticmethod
    def _sanitize(df: pd.DataFrame) -> pd.DataFrame:
        """Non-destructive sanitization: strip whitespace from column names."""
        df.columns = [str(c).strip() for c in df.columns]
        # Remove fully unnamed/empty columns introduced by trailing commas
        df = df.loc[:, ~df.columns.str.match(r"^Unnamed:")]
        return df


def load_titanic() -> pd.DataFrame:
    """Return the built-in Titanic dataset for demonstration purposes."""
    try:
        url = (
            "https://raw.githubusercontent.com/datasciencedojo/datasets"
            "/master/titanic.csv"
        )
        df = pd.read_csv(url)
        logger.info("Titanic dataset loaded from remote URL.")
        return df
    except Exception:
        logger.warning("Could not fetch Titanic dataset from network. Using bundled version.")
        return _bundled_titanic()


def _bundled_titanic() -> pd.DataFrame:
    """Minimal bundled Titanic dataset for offline use."""
    data = {
        "PassengerId": list(range(1, 21)),
        "Survived": [0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1],
        "Pclass": [3, 1, 3, 1, 3, 3, 1, 3, 3, 2, 3, 1, 3, 3, 3, 2, 3, 2, 3, 3],
        "Name": [f"Passenger {i}" for i in range(1, 21)],
        "Sex": ["male", "female", "female", "female", "male",
                "male", "male", "male", "female", "female",
                "female", "female", "male", "male", "female",
                "female", "male", "male", "female", "female"],
        "Age": [22, 38, 26, 35, None, None, 54, 2, 27, 14,
                4, 58, 20, 39, 14, 55, 2, None, 31, None],
        "SibSp": [1, 1, 0, 1, 0, 0, 0, 3, 0, 1, 1, 0, 0, 1, 0, 0, 4, 0, 1, 0],
        "Parch": [0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 1, 0, 0, 5, 0, 0, 1, 0, 0, 0],
        "Ticket": [f"T{i}" for i in range(1, 21)],
        "Fare": [7.25, 71.28, 7.92, 53.10, 8.05, 8.46, 51.86, 21.08,
                 11.13, 30.07, 16.70, 26.55, 8.05, 31.28, 7.85, 16.00,
                 29.13, 13.00, 18.00, 7.22],
        "Cabin": [None, "C85", None, "C123", None, None, "E46", None,
                  None, None, "G6", "C103", None, None, None, None,
                  None, None, None, None],
        "Embarked": ["S", "C", "S", "S", "S", "Q", "S", "S", "S", "C",
                     "S", "S", "S", "S", "S", "S", "Q", "S", "S", "C"],
    }
    return pd.DataFrame(data)
