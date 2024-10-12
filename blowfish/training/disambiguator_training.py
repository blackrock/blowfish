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

from pydantic import BaseModel, Field
from typing import List, ClassVar, Tuple
from sklearn.neighbors import KernelDensity
from sklearn.utils import shuffle

from blowfish.utils.constants import DEFAULT_KDE_FEATURES
from blowfish.calculations import calculate_relevant_features


class DisambiguationModelGenerator(BaseModel):
    module_name: ClassVar[str] = "DisambiguationGenerator"

    query_features_headers: List[str] = Field(default=["query_id"])

    kde_storage_path: str = Field(default="./")
    kde_storage_name: str = Field(default="disambiguator_kde.pkl")
    
    class Config:
        arbitrary_types_allowed = True
        extra = "ignore"
    

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def format_qa_eval_df(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Reformats the queries by flattening the evaluation results between a query and the top-k retrieved
        chunks.
        """
        expanded_qa_eval_df = dataframe[[
                                        "query_id",
                                        "chunk_topn_docname",
                                        "chunk_topn_scores",
                                        "topic_labels",
                                        "doc_match",
                                        "chunk_match",
                                        "query_embedding",
                                        "chunk_embeddings",
                                        "silhouette_score"
                                        ]]\
                                        .assign(topn_rank = lambda x: x["doc_match"].apply(lambda y: np.arange(0, len(y), 1)))\
                                        .explode([
                                                "chunk_topn_docname",
                                                "chunk_topn_scores",
                                                "topic_labels",
                                                "chunk_match",
                                                "doc_match",
                                                "silhouette_score",
                                                "chunk_embeddings",
                                                "topn_rank"
                                                ])\
                                        .rename(columns={
                                                        "chunk_topn_docname": "docname",
                                                        "chunk_topn_scores": "score",
                                                        "topic_labels": "label",
                                                        "topn_rank": "rank"
                                                        })
        expanded_qa_eval_df = expanded_qa_eval_df[expanded_qa_eval_df.label != -1].reset_index(drop=True)
        expanded_qa_eval_df["topic_label"] = [f"{dn}_{l}" for dn,l in zip(expanded_qa_eval_df["docname"].tolist(),
                                                                            expanded_qa_eval_df["label"].tolist())]
        
        return expanded_qa_eval_df


    def generate_queries_features(self, 
                                  dataframe: pd.DataFrame,
                                  kde_features_order: List[str] = DEFAULT_KDE_FEATURES) -> pd.DataFrame:
        """
            Generats the relevant features for the queries
        """
        queries_features = []
        for _, sub_df in dataframe.groupby("query_id"):
            sub_df = sub_df.sort_values(by="rank").drop_duplicates(["docname", "query_id", "score", "rank"])
            
            # skip if the query only has 1 retrieved chunk
            if len(sub_df) < 2:
                continue
            
            features = {
                        "query_id": sub_df.iloc[0]["query_id"],
                        "correct_prediction": int(sub_df.iloc[0]["chunk_match"])
                        }
            derived_features = calculate_relevant_features(sub_df, kde_features_order)
            features.update(derived_features)

            queries_features.append(pd.DataFrame(features, index=[0]))
            
        queries_features_df = pd.concat(queries_features).reset_index(drop=True)

        return queries_features_df
    

    def save_kde_model(self, kde: KernelDensity) -> None:
        """
            Saves a provided model as a pickle file
        """
        with open(self.kde_storage_path + self.kde_storage_name, "wb") as f:
            pickle.dump(kde, f)

        print(f"Saved trained disambiguator KDE to {self.kde_storage_path + self.kde_storage_name}")
        
        
    def __call__(self,
                qa_eval_results: pd.DataFrame,
                kde_features_order: List[str] = DEFAULT_KDE_FEATURES) -> Tuple[pd.DataFrame, KernelDensity]:        
        input_columns = set(qa_eval_results)
        expected_cols = {"query_id",
                         "chunk_topn_scores",
                         "doc_match",
                         "chunk_match",
                         "query_embedding",
                         "chunk_embeddings"}
        assert input_columns.issuperset(expected_cols)

        formatted_queries = self.format_qa_eval_df(qa_eval_results)
        queries_features = self.generate_queries_features(formatted_queries, kde_features_order)
        min_n_features = (lambda df: min(df[df["correct_prediction"]==0].__len__(),
                                         df[df["correct_prediction"]==1].__len__()))(queries_features)
        balanced_queries_features = shuffle((lambda df: pd.concat([df[df["correct_prediction"]==0].sample(min_n_features),
                                                                   df[df["correct_prediction"]==1].sample(min_n_features)]))
                                            (queries_features))
        KD = KernelDensity(bandwidth=0.2,kernel="gaussian")
        KD.fit(X = balanced_queries_features.drop(columns=["query_id"]))

        self.save_kde_model(kde=KD)
        
        return queries_features, balanced_queries_features, KD
        