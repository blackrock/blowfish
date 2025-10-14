from __future__ import annotations

import json
import os
from pathlib import Path
from typing import ClassVar, List, Tuple

import numpy as np
import pandas as pd
from faiss import IndexFlatL2, read_index, write_index
from pydantic import BaseModel, Field, field_validator


class FaissVDBIndexing(BaseModel):
    """
    Simple FAISS L2 indexer with JSON-backed ID mapping.

    - Persists the FAISS index to `vdb_path`
    - Persists the ID/hash-key mapping to `json_index_path` as {"map": [...]}
    - Validates vector dimensionality matches `vdb_vector_size`
    """

    module_name: ClassVar[str] = "FaissIndexer"

    vdb_path: str = Field(default="./faiss.index")
    json_index_path: str = Field(default="./index.json")
    vdb_reset_faiss_index: bool = Field(default=False)
    vdb_vector_size: int = Field(..., description="Embedding dimensionality")

    @field_validator("vdb_vector_size")
    @classmethod
    def _positive_dim(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("vdb_vector_size must be a positive integer.")
        return v

    def save(
        self,
        vector_index: IndexFlatL2,
        index_mapping: List[str],
    ) -> None:
        """
        Atomically save the FAISS index and the mapping JSON.
        """
        vdb_path = Path(self.vdb_path)
        json_path = Path(self.json_index_path)

        write_index(vector_index, str(vdb_path))

        tmp_json = json_path.with_suffix(json_path.suffix + ".tmp")
        with tmp_json.open("w", encoding="utf-8") as f:
            json.dump({"map": index_mapping}, f)
        tmp_json.replace(json_path)

    def load(self) -> Tuple[IndexFlatL2, List[str]]:
        """
        Load the FAISS index and mapping. If mapping file is missing but the
        FAISS index exists, initialize an empty mapping.
        """
        vdb_path = Path(self.vdb_path)
        json_path = Path(self.json_index_path)

        if not vdb_path.is_file():
            raise FileNotFoundError(f"FAISS index not found at {vdb_path}")

        vector_index = read_index(str(vdb_path))
        if not isinstance(vector_index, IndexFlatL2):
            # In case the stored index is not the expected type
            raise TypeError("Loaded FAISS index is not an IndexFlatL2.")

        if json_path.is_file():
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            index_mapping = list(data.get("map", []))
        else:
            index_mapping = []

        return vector_index, index_mapping

    def reset(self) -> Tuple[IndexFlatL2, List[str]]:
        """
        Create a fresh FAISS IndexFlatL2 with the configured dimensionality,
        empty mapping, and persist both.
        """
        vector_index = IndexFlatL2(self.vdb_vector_size)
        index_mapping: List[str] = []
        self.save(vector_index, index_mapping)
        return vector_index, index_mapping

    def load_or_init(self) -> Tuple[IndexFlatL2, List[str]]:
        """
        Load existing index/mapping unless reset is requested or index is missing.
        """
        if self.vdb_reset_faiss_index:
            return self.reset()

        if not Path(self.vdb_path).is_file():
            return self.reset()

        return self.load()

    def add_dataframe(
        self,
        df: pd.DataFrame,
        *,
        embedding_col: str = "chunk_embedding",
        id_col: str = "hash_key",
    ) -> Tuple[IndexFlatL2, List[str]]:
        """
        Add vectors + IDs from a DataFrame and persist.

        Expects:
          - `embedding_col` column containing 1D array-like embeddings
          - `id_col` column containing string IDs (e.g., hash keys)
        """
        vector_index, index_mapping = self.load_or_init()

        try:
            vectors = np.vstack(df[embedding_col].to_list()).astype(np.float32)
        except KeyError:
            raise KeyError(f"Missing required column '{embedding_col}'.")
        except Exception as e:
            raise ValueError(f"Failed to build vector matrix: {e}")

        if vectors.ndim != 2 or vectors.shape[1] != self.vdb_vector_size:
            raise ValueError(
                f"Vector shape mismatch: got {vectors.shape}, "
                f"expected (*, {self.vdb_vector_size})."
            )

        try:
            new_ids = df[id_col].astype(str).to_list()
        except KeyError:
            raise KeyError(f"Missing required column '{id_col}'.")

        vector_index.add(vectors)

        index_mapping.extend(new_ids)

        self.save(vector_index, index_mapping)
        return vector_index, index_mapping

    def __call__(self, input: pd.DataFrame) -> Tuple[IndexFlatL2, List[str]]:
        """
        Backwards-compatible entrypoint:
        expects DataFrame with 'chunk_embedding' and 'hash_key' columns.
        """
        return self.add_dataframe(input)
