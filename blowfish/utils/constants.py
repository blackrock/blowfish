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

DEFAULT_KDE_FEATURES = ["scale_mean",
                        "scale_min",
                        "iq25-75_scale",
                        "max_homology_birth",
                        "mean_homology_birth",
                        "std_homology_birth",
                        "mean_homology1st_birth",
                        "mean_homology1st_lifetime",
                        "top_k_doc_spread",
                        "top_k_topic_spread",
                        "silhouette_score_mean",
                        "silhouette_score_std"]