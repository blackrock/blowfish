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
from sentence_transformers import SentenceTransformer
from langchain_openai.embeddings.azure import AzureOpenAIEmbeddings
from langchain_openai.embeddings import OpenAIEmbeddings

class OpenAIEmbeddingWrapper(OpenAIEmbeddings):
    def encode(self, documents: List[str]) -> np.ndarray:
        results = np.array(self.embed_documents(documents))
        return results

class AZOpenAIEmbeddingWrapper(AzureOpenAIEmbeddings):
    def encode(self, documents: List[str]) -> np.ndarray:
        results = np.array(self.embed_documents(documents))
        return results

class EmbeddingModelHooks():
    sentence_transformer = SentenceTransformer
    azure_openai = AZOpenAIEmbeddingWrapper
    openai = OpenAIEmbeddingWrapper
        