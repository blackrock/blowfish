"""
Copyright 2024 BlackRock, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import pandas as pd
import pickle
import httpx
from pydantic import BaseModel, Field
from typing import List, ClassVar, Any, Union
from blowfish.utils.embedding_models_factory import EmbeddingModelHooks


class BulkQueriesEmbedder(BaseModel):
    
    module_name: ClassVar[str] = "BulkQueriesEmbedder"
   
    disable_ssl: bool = Field(default=False)
    llm_encoder_config: dict = Field()
    llm_encoder_type: str = Field(default="sentence_transformer")
    
    LLM_encoder: Any = None
    
    class Config:
        arbitrary_types_allowed = True
        extra = "ignore"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        llm_encoder_class = getattr(EmbeddingModelHooks,
                                   self.llm_encoder_type)
        if self.disable_ssl:
            self.LLM_encoder = llm_encoder_class(**self.llm_encoder_config, http_client=httpx.Client(verify=False))
        else:
            self.LLM_encoder = llm_encoder_class(**self.llm_encoder_config)
        
    def __call__(self, input: pd.DataFrame, save_file: str = "queries_embeddings") -> pd.DataFrame:
        try:
            input_columns = set(input)
            expected_cols = {"query", "docname", "answer"}
            assert input_columns.issuperset(expected_cols)
            
            query_embeddings = self.LLM_encoder.encode(input["query"].to_list())
            input["query_embedding"] = list(query_embeddings)
            input["query_id"] = input.index

            with open(f"{save_file}.pkl","wb") as f:
                pickle.dump(input, f)
        
            return input
            
        except AssertionError:
            print(f"Error parsing query.\n Missing columns: {expected_cols.difference(input_columns)}")
            