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

from typing import Any, Dict, List
import numpy as np
import pandas as pd
from gtda.homology import VietorisRipsPersistence


def calculate_scaled_distance_distribution(sub_df: pd.DataFrame, *args) -> Dict[str, Any]:
    scale = (np.sort(sub_df.score) / np.sort(sub_df.score)[0])[1:]
    
    scale_distribution = {
                        "scale_mean": np.mean(scale),
                        "scale_min": np.min(scale),
                        "iq25-75_scale": np.quantile(scale, 0.75) - np.quantile(scale, 0.25),
                        }
    return scale_distribution

def calculate_first_order_homology_distribution(sub_df: pd.DataFrame, *args) -> Dict[str, Any]:
    vietorisRipsGenerator = VietorisRipsPersistence(homology_dimensions=(0,1))

    chunks_embed = np.array(sub_df["chunk_embeddings"].to_list())
    query_embed = np.array(sub_df["query_embedding"].to_list())

    renormalised_embeddings = chunks_embed - query_embed
    renormalised_embeddings = renormalised_embeddings / np.linalg.norm(renormalised_embeddings, axis=1)[:,np.newaxis]

    diagrams = vietorisRipsGenerator.fit_transform(renormalised_embeddings[:,:][np.newaxis,:,:])
    
    neighbour_0th_homology = diagrams[diagrams[:, :, 2] == 0,:][:, 1]
    neighbour_1st_homology = diagrams[diagrams[:, :, 2] == 1,:][:, :2]

    holes_lifetimes = (neighbour_1st_homology[:, 1]-neighbour_1st_homology[:, 0])
            
    homology_distribution = {
                            "max_homology_birth": np.max(neighbour_0th_homology),
                            "mean_homology_birth": np.mean(neighbour_0th_homology),
                            "std_homology_birth": np.std(neighbour_0th_homology),
                            "mean_homology1st_birth": np.mean(neighbour_1st_homology,axis=0)[0],
                            "mean_homology1st_lifetime": np.mean(holes_lifetimes),
                            }
    return homology_distribution


def calculate_silhouette_score_distribution(sub_df: pd.DataFrame, *args) -> Dict[str, Any]:
    silhouette_score_distribution = {
                                    "silhouette_score_mean": sub_df["silhouette_score"].mean(),
                                    "silhouette_score_std": sub_df["silhouette_score"].std()
                                    }
    return silhouette_score_distribution


def calculate_doc_spread(sub_df: pd.DataFrame) -> Dict[str, Any]:
    return {"top_k_doc_spread": len(set(sub_df["docname"])) / len(sub_df)}


def calculate_topic_spread(sub_df: pd.DataFrame) -> Dict[str, Any]:
    return {"top_k_topic_spread": len(set(sub_df["topic_label"])) / len(sub_df)}


def calculate_relevant_features(sub_df: pd.DataFrame, kde_features_order: List[str]) -> Dict[str, Any]:
    feature_set = set(kde_features_order)

    features = calculate_scaled_distance_distribution(sub_df)
    features.update(calculate_first_order_homology_distribution(sub_df))
    features.update(calculate_silhouette_score_distribution(sub_df))
    features.update(calculate_doc_spread(sub_df))
    features.update(calculate_topic_spread(sub_df))

    return {k: v for k, v in features.items() if k in feature_set}
