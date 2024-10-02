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

import numpy as np
import pandas as pd
import pickle
import httpx
from typing import List, ClassVar, Any
from pydantic import BaseModel, Field
from ast import literal_eval
from blowfish.utils.embedding_models_factory import EmbeddingModelHooks
import pathlib


class NaiveChunksEmbedding(BaseModel):

    module_name: ClassVar[str] = "NaiveChunkEmbedder"

    disable_ssl: bool = Field(default=False)
    llm_encoder_config: dict = Field()
    llm_encoder_type: str = Field(default="sentence_transformer")

    embeddings_storage_dir: str = Field(default='./')
    
    LLM_encoder: Any = None
    
    """
        Pydantic Config
    """
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

    def generate_LC_embeddings(self,
                               dfs: pd.DataFrame,
                               **kwargs):
        
        chunks_df = dfs
        chunks_df["chunk_embedding"] = list(self.LLM_encoder.encode(chunks_df["Text"].to_list(),**kwargs))
            
        return chunks_df

    def get_chunk_hash_key(self, 
                           dataframe: pd.DataFrame
                           ) -> pd.DataFrame:
        keys = [f"{d}_{idx}" for d, idx in zip(dataframe["docname"],
                                                 dataframe["chunk_index"])]
        dataframe["hash_key"] = keys
        return dataframe
    
    def save_embeddings_df(self, 
                           dataframe: pd.DataFrame,
                           docname: str
                           ) -> None:

        with open(self.embeddings_storage_dir + docname + "_chunk_embeddings.pkl","wb") as f:
            pickle.dump(dataframe, f)        
    
    def __call__(self,
                 input: pd.DataFrame,
                 docname: str = 'document',
                 **kwargs) -> pd.DataFrame:
        
        if 'chunk_index' not in input.columns:
            input = input.assign(chunk_index=np.arange(0,len(input),1))
            
        if 'hash_key' not in input.columns:
            input = self.get_chunk_hash_key(input)
        
        chunks_embeddings_df = self.generate_LC_embeddings(input, **kwargs)
        chunks_embeddings_df = chunks_embeddings_df[['docname',
                                                   'Text',
                                                   'chunk_embedding',
                                                   'hash_key'
                                                   ]]

        self.save_embeddings_df(chunks_embeddings_df, docname)

        return chunks_embeddings_df