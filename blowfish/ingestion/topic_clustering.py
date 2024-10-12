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
import traceback
import pickle

from typing import List, ClassVar, Any
from pydantic import BaseModel, Field
from umap import UMAP
from hdbscan import HDBSCAN
from tqdm import tqdm

from sklearn.metrics import silhouette_score, silhouette_samples


class TopicClusterGenerator(BaseModel):
    
    module_name: ClassVar[str] = "TopicClusterer"
    
    topics_storage_dir: str = Field(default= "./")
    
    def detect_umap_nneighbours_optimum(self,
                                        dataframe: pd.DataFrame,
                                        range: np.ndarray = np.arange(2,40,2),
                                        tolerance: int = 0.1) -> pd.DataFrame:
        cumsums = []
        projections = []
        for u in tqdm(range):
            umap_engine = UMAP(n_neighbors=u, metric="cosine")
            projection = umap_engine.fit_transform(np.array(dataframe["chunk_embedding"].to_list()))
            projections.append((projection, umap_engine))
            h,_ = np.histogram(np.sqrt(np.sum((projection[:,np.newaxis,:]-projection[np.newaxis,:,:])**2,axis=-1)).flatten(),
                               bins=np.arange(0,50,0.1))
            cumsums.append(np.cumsum(h))
            
        cumsums = np.array(cumsums)
        cumsums -= cumsums[0,:]
        var_cumsum = np.diff(cumsums.sum(1))
        opt_umap_nn = [0]
        for i,_ in enumerate(var_cumsum):
            print(f"{var_cumsum[i:i+3].mean()}, {var_cumsum[i]}, {tolerance * (var_cumsum[:i+1].sum())}")
            if (var_cumsum[i:i+3].mean()) > tolerance * (var_cumsum[:i+1].sum()):
                opt_umap_nn += [i+1]
            else:
                if i == 0:
                    continue
                break
        print(f"optimum nearest neigbours is {range[opt_umap_nn[-1]]}")
        dataframe = dataframe.assign(X=projections[opt_umap_nn[-1]][0][:,0])
        dataframe = dataframe.assign(Y=projections[opt_umap_nn[-1]][0][:,1])
        
        return dataframe
    
    def run_hdbscan(self, dataframe: pd.DataFrame) -> List[int]:
        """
            Performs HDBScan to cluster topics
        """
        shscore = []
        outliers = []
        skipped_min_samples = []
        X = np.arange(2,100,1)
        max_cluster_size = np.round(len(dataframe)/2.0).astype(int)
        for i in np.arange(2,100,1):
            try:
                myhdbscan = HDBSCAN(min_cluster_size=i,
                                    max_cluster_size=max_cluster_size,
                                    gen_min_span_tree=True, 
                                    min_samples=1)
                labels = myhdbscan.fit_predict(dataframe[['X', 'Y']].to_numpy())
                projections_xy = dataframe[['X', 'Y']].to_numpy()[labels>=0,:]
                if projections_xy.shape[0] == 0:
                    skipped_min_samples.append(i)
                    continue
                labels_no_outliers = labels[labels>=0]
                if len(set(labels_no_outliers)) < 2:
                    continue
                outliers.append(sum(labels==-1)/len(labels))
                shscore.append(silhouette_score(projections_xy[:,:],labels=labels_no_outliers[:]))
            except Exception:
                print(f"HDBSCAN ERRROR:{traceback.format_exc()}; stopped at iteration {i}")
                break
        print(f"Skipped min samples: {skipped_min_samples}")
        opt_cluster_size = X[np.argmax(np.array(shscore) - np.array(outliers))]
        print(f"Sscores: {shscore}", opt_cluster_size)
        
        myhdbscan = HDBSCAN(min_cluster_size=opt_cluster_size,
                            max_cluster_size=max_cluster_size,
                            gen_min_span_tree=True,
                            min_samples=1)
        labels = myhdbscan.fit_predict(dataframe[['X', 'Y']].to_numpy())
        
        return list(labels)

    def calculate_sihouette_scores(self, 
                                   dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe = dataframe.assign(silhouette_score = silhouette_samples(dataframe[['X', 'Y']].to_numpy(),
                                                                           labels=dataframe.label.to_list()))
        
        return dataframe

    def cluster_datapoints(self,
                           dataframe: pd.DataFrame, 
                            ) -> pd.DataFrame:
        """
            Clusteres the topics with th optimal settings
        """
        try:
            projected_df = self.detect_umap_nneighbours_optimum(dataframe)
            projected_df = projected_df.assign(label=self.run_hdbscan(projected_df))
            projected_df = self.calculate_sihouette_scores(projected_df)
            
        except Exception:
            print(traceback.format_exc())
   
        return projected_df
    
    def save_topics_df(self,
                       dataframe: pd.DataFrame,
                       docname: str = 'document') -> None:
        """
            Saves the generated topics dataframe into a pickle file
        """
        with open(self.topics_storage_dir + docname + "_chunk_topics.pkl","wb") as f:
            pickle.dump(dataframe, f)    
    
    def __call__(self,
                 input: pd.DataFrame,
                 docname: str) -> pd.DataFrame:
        
        df_with_clusters = self.cluster_datapoints(input)
        self.save_topics_df(df_with_clusters,docname)
        
        return df_with_clusters