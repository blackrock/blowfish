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
from sklearn.neighbors import KernelDensity
from typing import List, Tuple

from blowfish.utils.constants import DEFAULT_KDE_FEATURES
from blowfish.calculations.calculations import calculate_relevant_features

import warnings
warnings.filterwarnings('ignore')

class AmbiguityScorer():
    def __init__(self, 
                 kde: KernelDensity,
                 topics_df: pd.DataFrame,
                 indexes = ["docname"]):
        self.indexes = indexes
        self.ProbaDensityKernel = kde
        topics = topics_df
        self.topics = self.format_topics(topics)
        
    def format_topics(self, topics_df: pd.DataFrame) -> pd.DataFrame:
        try:
            required_columns = ["docname", "chunk_embedding", "label", "silhouette_score", "hash_key"]
            assert all([c in topics_df.columns for c in required_columns])

            topics = topics_df.assign(cluster_id=  [f"{d}_{l}" for d,l in zip(topics_df.docname.tolist(),topics_df.label.tolist())])
            topics = topics.set_index(self.indexes)
            return topics
        
        except AssertionError:
            print(f"topics_df not formatted correctly, need {required_columns}\n but has {topics_df.columns}")
            return None
    
    def join_topics_to_query_and_chunks(self, 
                                        queries_results: pd.DataFrame,
                                        columns_of_interest: list=["topn_docname",
                                                                   "topn_scores",
                                                                   "topn_rank",
                                                                   "query_embedding",
                                                                   "chunk_embeddings",
                                                                   "hash_key"]) -> pd.DataFrame:
        try:
            assert all([c in queries_results.columns for c in columns_of_interest])

            queries_results = queries_results[columns_of_interest]\
                                        .rename(columns={"topn_docname": "docname",
                                                        "topn_scores": "score",
                                                        "topn_rank": "rank"})
            
            query_with_associated_topics = pd.merge(queries_results, self.topics[["label", "silhouette_score", "hash_key"]], on='hash_key', how="left")
            query_with_associated_topics = query_with_associated_topics[query_with_associated_topics.label != -1].reset_index()
            query_with_associated_topics["topic_label"] = [f"{dn}_{l}" for dn,l in zip(query_with_associated_topics["docname"].tolist(),
                                                                                query_with_associated_topics["label"].tolist())]
            return query_with_associated_topics
        
        except AssertionError:
            print(f"Dataframe not formatted correctly, need {columns_of_interest}\n but has {queries_results.columns}")
            return None
    
    def get_correctness_probability(self, sample: List[np.float64]) -> np.float64:
        sample_correct = [1] + sample
        sample_incorrect = [0] + sample

        sample_correct = sample_correct[:self.ProbaDensityKernel.n_features_in_]
        sample_incorrect = sample_incorrect[:self.ProbaDensityKernel.n_features_in_]

        proba = np.exp(self.ProbaDensityKernel.score_samples([sample_correct]))/\
            (np.exp(self.ProbaDensityKernel.score_samples([sample_correct])) + np.exp(self.ProbaDensityKernel.score_samples([sample_incorrect])))
        
        return proba
        
    def calculate_query_correctness_probability(self, query_df: pd.DataFrame, kde_features_order: List[str] = DEFAULT_KDE_FEATURES) -> pd.DataFrame:
        sample = {}
        features = calculate_relevant_features(query_df, kde_features_order)
        input_samples = [features[feature] for feature in kde_features_order]    # This ensures order

        p_correct = self.get_correctness_probability(input_samples)
        
        sample["p_correct"] = p_correct
        sample.update(features)

        return pd.DataFrame(sample)
        
    def run_scoring(self, query_top_hits_df: pd.DataFrame) -> Tuple[float, pd.DataFrame, pd.DataFrame]:
        """ This function calculates the query correctness probability
        :param query_top_hits_df: (pd.DataFrame) query features engineered from its localization in embedding space with respect to k-nearest neighbours 
        :return: (tuple) correctness probability, correctness features, nearest neighbours 
        """
        query_with_associated_topics = self.join_topics_to_query_and_chunks(query_top_hits_df)
        sample = self.calculate_query_correctness_probability(query_with_associated_topics)
        
        return sample.p_correct.iloc[0], sample, query_with_associated_topics
