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

from typing import List, ClassVar, Any, Tuple
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from collections import defaultdict
from tqdm import tqdm
import Levenshtein
import traceback
from blowfish.utils.vdb_factory import VDBHooks

class BulkQueriesEvaluator(BaseModel):
    
    module_name: ClassVar[str] = "BulkQueriesEvaluator"

    top_k_results: int = Field()
    vdb_path: str = Field(default="./faiss.index")
    vdb_type: str = Field()
    
    vdb_index: Any = None
    vdb_mapping: Any = None
    
    class Config:
        arbitrary_types_allowed = True
        extra = "ignore"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        index, idx_mapping = getattr(VDBHooks, self.vdb_type)(**kwargs).load_index()
        self.vdb_index = index
        self.vdb_mapping = idx_mapping
        
    
    def evaluate_isin(self,
                      query_series: pd.Series, 
                      answer: pd.Series) -> Tuple[bool]:
        
        docname, content = answer[["docname", "Text"]]
        
        isindocument = query_series.docname == docname
        isinchunk = Levenshtein.distance(content,query_series.answer,weights=(0,1,0)) == 0
        
        return isindocument, isindocument*isinchunk
    

    def __call__(self, queries_df: pd.DataFrame, topics_df: pd.DataFrame) -> pd.DataFrame:
        """ Performs a vector db retrieval and compares the retrieved chunks with the original chunk """
        
        try:  
            assert isinstance(queries_df, pd.DataFrame)
            assert isinstance(topics_df, pd.DataFrame)
            
            input_columns = set(queries_df)
            expected_cols = {"query", "docname", "query_embedding", "answer"}
            assert input_columns.issuperset(expected_cols)
            
            input_columns = set(topics_df)
            expected_cols = {'docname', 'Text', 'chunk_embedding', 'hash_key','X', 'Y', 'label', 'silhouette_score'}
            assert input_columns.issuperset(expected_cols)
            
            topics_df = topics_df.set_index("hash_key")
            
            QA_output = []
            
            for _, row in tqdm(queries_df.iterrows()):
                query_results = defaultdict(list)

                Qembed = np.array(row.query_embedding)[np.newaxis,:]
                D, I = self.vdb_index.search(Qembed, self.top_k_results)  # search

                for score, fidx in zip(D[0],I[0]):
                    hash_key = self.vdb_mapping[fidx]
                    retrieved_chunk = topics_df.loc[hash_key]
                    query_results["chunk_topn_answer"].append(retrieved_chunk["Text"])
                    query_results["chunk_topn_scores"].append(score)
                    query_results["chunk_topn_docname"].append(retrieved_chunk["docname"])
                    query_results["topic_labels"].append(retrieved_chunk['label'])
                    query_results["chunk_embeddings"].append(retrieved_chunk["chunk_embedding"])
                    dmatch, cmatch = self.evaluate_isin(row, retrieved_chunk)
                    query_results["doc_match"].append(dmatch)
                    query_results["chunk_match"].append(cmatch)
                    query_results["silhouette_score"].append(retrieved_chunk["silhouette_score"])
                    
                QA_output.append(pd.concat([row, pd.Series(query_results)]))
            return pd.DataFrame(QA_output)
        
        except AssertionError:
            print(f"Error parsing query.\n Missing columns: {expected_cols.difference(input_columns)}")
            print(f"Error position: {traceback.format_exc()}")
            return None