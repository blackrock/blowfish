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
import os
import json
from faiss import write_index, read_index, IndexFlatL2
from typing import List, ClassVar, Any, Tuple
from pydantic import BaseModel, Field

from blowfish.utils.embedding_models_factory import EmbeddingModelHooks
   
class FaissVDBIndexing(BaseModel):
    
    module_name: ClassVar[str] = "FaissIndexer"
    
    vdb_path: str = Field(default="./faiss.index")
    json_index_path: str = Field(default="./index.json")
    vdb_reset_faiss_index: bool = Field(default=False)
    vdb_vector_size: int = Field()
    
    def save_index(self,
                   vector_index: IndexFlatL2,
                   index_mapping: list):
        write_index(vector_index, self.vdb_path)
        with open(self.json_index_path, "w") as f:
            json.dump({"map": index_mapping}, f)
            
    def load_index(self):
        vector_index = read_index(self.vdb_path)
        with open(self.json_index_path, "r") as f:
            index_mapping = json.load(f)["map"]
        return vector_index, index_mapping
                
    def reset_index(self):
        vector_index = IndexFlatL2(self.vdb_vector_size)
        index_mapping = []
        self.save_index(vector_index,index_mapping)
        return vector_index, index_mapping
    
    
    def __call__(self, 
                 input: pd.DataFrame) -> Tuple:
        
        if not os.path.isfile(self.vdb_path):
            vector_index, index_mapping = self.reset_index()
        elif self.vdb_reset_faiss_index:
            vector_index, index_mapping = self.reset_index()
        else: 
            vector_index, index_mapping = self.load_index()
            
            
        vectors = np.array(input["chunk_embedding"].to_list())
        hash_keys = input["hash_key"].to_list()
        vector_index.add(vectors)
        index_mapping += hash_keys
        self.save_index(vector_index, index_mapping)
        
        return vector_index, index_mapping
