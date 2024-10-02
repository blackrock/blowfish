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

from typing import List
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
import shap
from blowfish.utils.constants import DEFAULT_KDE_FEATURES

import warnings
warnings.filterwarnings('ignore')

class FeedbackDecider():
    
    def __init__(self, kde: KernelDensity, kde_features_order: List[str] = DEFAULT_KDE_FEATURES):
        self.KDE = kde
        self.high_accuracy_samples = self.get_high_accuracy_samples()
        self.explainer = shap.KernelExplainer(self.KDE_prediction, self.high_accuracy_samples)
        self.KDE_features_order = kde_features_order
        
    def get_high_accuracy_samples(self, num_samples: int = 50) -> np.array:
        """Obtain [num_samples] samples from KDE"""
        feat_samples = []

        while len(feat_samples) < num_samples:
            sample = self.KDE.sample(n_samples=1)
            if np.round(sample[0][0]) == 1.0:
                neg_sample = np.copy(sample)
                pos_sample = np.copy(sample)
                neg_sample[0,0] = 0.0
                pos_sample[0,0] = 1.0

                proba = np.exp(self.KDE.score(sample))/(np.exp(self.KDE.score(sample)) + np.exp(self.KDE.score(neg_sample)))
                if proba > 0.8:
                    feat_samples.append(sample[0][1:])

        return np.array(feat_samples)
    
    def KDE_prediction(self, sample):
        X_test = np.ones(sample.shape[0])
        X_test0 = 1 - X_test
        proba = np.exp(self.KDE.score_samples(np.concatenate([X_test[:,np.newaxis], sample],axis=1)))/\
                (np.exp(self.KDE.score_samples(np.concatenate([X_test[:,np.newaxis], sample],axis=1)))
                 + np.exp(self.KDE.score_samples(np.concatenate([X_test0[:,np.newaxis], sample],axis=1))))
        return proba
    
    def explain_query(self, 
                      query_features: pd.DataFrame, 
                      relevance_threshold: float = 0.1) -> str:
        """ This function calculates SHAP values for query and generate feedback prompt that depends on feature importance

        :param query_features: (pd.Series) query features engineered from its localization in embedding space with respect to k-nearest neighbours 
        :param relevance_threshold: (float) feature relevance threshold. Below, not considered for prompt generation
        :return: (str) new prompt generator type
        """
        
        query_features_array = query_features[self.KDE_features_order].to_numpy()
        query_shap_values = self.explainer(query_features_array)
        query_shap_values.feature_names = self.KDE_features_order
        
        query_shap_df = pd.DataFrame(dict(**{k:v for k,v in query_shap_values[0,:].__dict__["_s"].__dict__.items() if k in ["values","data"]},
                          **{"features":self.KDE_features_order})).sort_values(by="values")
        query_shap_df = query_shap_df.sort_values(by="values")
        query_shap_df = query_shap_df[query_shap_df["values"] < 0.0]
        query_shap_df["fraction"] = query_shap_df["values"]/query_shap_df["values"].sum()
        
        relevant_features = query_shap_df[query_shap_df["fraction"] > relevance_threshold]["features"].tolist()
        
        if "top_k_topic_spread" in relevant_features:
            return "topicspread"
        elif "docspread" in relevant_features:
            return "top_k_doc_spread"
        else:
            return "dataspread"
            