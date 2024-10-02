# Blowfish

Blowfish is a library used to train a Gaussian KDE model to quantify and measure ambiguity in semantic search and also to perform inference on questions using the trained model.
This is the official implementation of the research published by Thomas R. Barillot and Alex De Castro from BlackRock - [Blowfish: Topological and statistical signatures for quantifying ambiguity in semantic search](https://arxiv.org/abs/2406.07990).

## Table of Contents

- [Blowfish](#blowfish)
  - [Table of Contents](#table-of-contents)
  - [Abstract](#abstract)
  - [Installation](#installation)
  - [Project Structure](#project-structure)
  - [Usage](#usage)
    - [Ingestion](#ingestion)
    - [Training](#training)
    - [Inference](#inference)
  - [Model Support](#model-support)
  - [Configurations](#configurations)
    - [Required Parameters](#required-parameters)
    - [Optional Parameters](#optional-parameters)
  - [KDE Features](#kde-features)
  - [Example Inputs Visualized](#example-inputs-visualized)
  - [Contribution](#contribution)
  - [License](#license)
  - [Credits](#credits)
  - [Citation](#citation)

## Abstract

This works reports evidence for the topological signatures of ambiguity in sentence embeddings that could be leveraged for ranking and/or explanation purposes in the context of vector search and Retrieval Augmented Generation (RAG) systems. We proposed a working definition of ambiguity and designed an experiment where we have broken down a proprietary dataset into collections of chunks of varying size - 3, 5, and 10 lines and used the different collections successively as queries and answers sets. It allowed us to test the signatures of ambiguity with removal of confounding factors. Our results show that proxy ambiguous queries (size 10 queries against size 3 documents) display different distributions of homologies 0 and 1 based features than proxy clear queries (size 5 queries against size 10 documents). We then discuss those results in terms increased manifold complexity and/or approximately discontinuous embedding sub-manifolds. Finally we propose a strategy to leverage those findings as a new scoring strategy of semantic similarities.

## Installation

### Prerequisites

```bash
pip install -r requirements.txt
```

### From Source

```bash
pip install -e .
```

## Project Structure

This project is divided into 3 sections, each handling the specified tasks.

* **Ingestion**
  * Chunk Embedding
  * Topic Modelling
  * Vector DB Indexing
* **Training**
  * Query Embedding
  * Chunk Retrieval Evaluation
  * KDE Training
* **Inference**
  * Ambiguity Scoring
  * Feature Decider

## Usage

### Ingestion

For more detailed information on the configurations please refer to the [configurations section](#configurations).
The docname parameter is an optional parameter in the embedder and clusterer step and is used as a filename where the embedded chunks and topics will be saved to. The default value of the parameter is **_document_**, the resulting save file will look like `{docname}_chunk_embeddings.pkl` and `{docname}_chunk_topics.pkl` respectively.
**Important:** Ensure that your hash_key is unique! This will be used during the inference step to map the chunks to the correct topic features.

```py
import pandas

from blowfish.ingestion.chunk_embeddings import NaiveChunksEmbedding
from blowfish.ingestion.topic_clustering import TopicClusterGenerator
from blowfish.ingestion.vdb_indexing import FaissVDBIndexing

# AzureOpenAIEmbeddings Configuration Example
azure_openai_config = {
                        "deployment": <model_name>,
                        "openai_api_key": <token>,
                        "azure_endpoint": <endpoint>,
                        "openai_api_version": <version>
                        }

# Sentence Transformers Configuration Example:
sentence_transformer_config = {
                               "model_name_or_path": "sentence-transformers/all-mpnet-base-v2"
                               # Add other sentence transformers config here
                               }

# Configuration Example Using a Sentence Transformer Model
configuration = {
                  "llm_encoder_config": sentence_transformer_config,
                  "vdb_vector_size": 768,
                  "llm_encoder_type": "sentence_transformer",
                  "top_k_results": 5,
                  "vdb_type": "faiss_index"
                  }

# Ingestion Module
chunks_embedder = NaiveChunksEmbedding(**configuration)
clusterer = TopicClusterGenerator()
indexer = FaissVDBIndexing(**configuration)

# load your dataframe containing columns: ['Text', 'docname', 'hash_key']
chunks_df = pd.read_csv(...)

embedded_chunks = chunks_embedder(chunks_df, docname='example_doc')
topics_df = clusterer(embedded_chunks, docname='example_doc')
vector_index, vector_mapping = indexer(topics_df)
```

### Training

The training data should be a Q&A set where the answer will be the chunk that the question was derived from. (Question & Source Chunk might be a better name here)
The list of possible features are listed in the [KDE Features section](#kde-features).
**Important**: ensure that the value in the docname field in your Q&A training set matches the value in the docname field in the topics dataframe. The query evaluation process will use string comparison to match these strings. Different values (e.g. `document1.pdf` vs `document1`) will result in incorrect evaluation.

```py
from blowfish.training.queries_embeddings import BulkQueriesEmbedder
from blowfish.training.queries_evaluation import BulkQueriesEvaluator
from blowfish.training.disambiguator_training import DisambiguationModelGenerator

# Training Module
queries_embedder = BulkQueriesEmbedder(**configuration)
queries_evaluator = BulkQueriesEvaluator(**configuration)
model_generator = DisambiguationModelGenerator(**configuration)

# load your training data containing columns: ['query', 'answer', 'docname']
qa_training_data = pd.read_csv(...)

embedded_queries = queries_embedder(qa_training_data)
query_eval = queries_evaluator(embedded_queries, topics_df)  # topics_df should be the generated topics from the ingestion step
queries_features, balanced_queries_features, kernel_density = model_generator(query_eval)  # kde will be saved locally as a .pkl file in this step
```

### Inference

The **full topics dataframe** and the **KDE model** is required for inference. If you have different topics dataframe as a result of multiple documents from the ingestion step, be sure to combine them together to get the full topics dataframe.

The clarity score obtained from `scorer.run_scoring(features_df)` represents the ambiguity or answerability of the question given the chunks. A higher score is better.

The decider uses [SHapley Additive exPlanations](https://shap.readthedocs.io/en/latest/) to determine which features contribute the most to the clarity scores. The idea behind the decider is gain visibility into what is causing the ambiguity and we leave this section up to the user on how they would like to utilize this information.

The output from the decider will be one of: "topicspread", "docspread", and "dataspread".

"topicspread" measures the number of unique topics the retrieved chunks contain. This topic of a chunk is determined from the clustering step.

"docspread", short for document spread, measures the number of documents the retrieved chunks come from.

"dataspread" is returned if other features such as the topological features contributes the most to the ambiguity.

```py
import pickle
from blowfish.inference.scorer import AmbiguityScorer
from blowfish.inference.decider import FeedbackDecider

kde = pickle.load(...)         # Load saved KDE from training step
topics_df = pickle.load(...)   #  Load saved topics Dataframe from ingestion step

scorer = AmbiguityScorer(kde, topics_df)
decider = FeedbackDecider(kde)

""" 
  Create features df from the query and the data from the retrieval step
  An example of this df could be found in the [Example Inputs Visualized] section below
  The features df must contain the columns:
  ['topn_docname', 'topn_scores', 'topn_rank', 'query_embedding', 'chunk_embeddings', 'hash_key']
"""
features_df = ...   # the combined dataframe
clarity_score, query_features, chunks_with_topics =  scorer.run_scoring(features_df)

# Obtain most relevant feature contributing to clarity score
explanation = decider.explain_query(query_features)
```

## Model Support

We currently support embeddings models with the following classes out of the box:

* SentenceTransformer
* OpenAIEmbeddings
* AzureOpenAIEmbeddings

If you want to use custom models, navigate to _[blowfish/utils/embedding_models_factory.py](/blowfish/utils/embedding_models_factory.py)_ and add your own class there that implements the `encode()` function. The encode function should return a numpy array of the embeddings.

## Configurations

### Required Parameters

| Name                       | Description                                                                                  |
| :------------------------- | :------------------------------------------------------------------------------------------- |
| llm_encoder_config         | configuration to instantiate LLM from base class                                             |
| llm_encoder_type           | wrapper name from [embedding_models_factory.py](/blowfish/utils/embedding_models_factory.py) |
| top_k_results              | Number of chunks retrieved by retriever                                                      |
| vdb_type                   | vector db type (only faiss is supported at the moment)                                       |
| vdb_vector_size            | Size of embeddings generated by model                                                        |

### Optional Parameters

| Name                       | Description                                                                 |
| :------------------------- | :-------------------------------------------------------------------------- |
| embeddings_storage_dir     | directory where chunk embeddings df will be saved (defaults to `./`)        |
| topics_storage_dir         | directory where topics df will be saved (defaults to `./`)                  |
| vdb_path                   | Path where vdb is saved (defaults to `./faiss.index`)                       |
| json_index_path            | Path where json index is saved (defaults to `./index.json`)                 |
| vdb_reset_faiss_index      | Whether to overwrite existing (defaults to `False`)                         |
| kde_storage_name           | name of file where KDE will be stored (defaults to `disambiguator_kde.pkl`) |
| disable_ssl                | disables ssl ~ used for OpenAI Models (defaults to `False`)                 |

## KDE Features

| Feature name              |
| :------------------------ |
| scale_mean                |
| scale_min                 |
| iq25-75_scale             |
| max_homology_birth        |
| mean_homology_birth       |
| std_homology_birth        |
| mean_homology1st_birth    |
| mean_homology1st_lifetime |
| top_k_docspread           |
| top_k_topic_spread        |
| silhouette_score_mean     |
| silhouette_score_std      |

## Example Inputs Visualized

### [chunks_df](#ingestion)

The information here is sourced from the respective wiki pages for the plants [Iris](https://en.wikipedia.org/wiki/Iris_(plant)) & [Osmanthus](https://en.wikipedia.org/wiki/Osmanthus)

| Text                                                                                                                             | docname           | hash_key |
| :------------------------------------------------------------------------------------------------------------------------------- | :---------------- | :------- |
| Nearly all Iris species are found in temperate Northern Hemisphere zones, from Europe to Asia and across North America.          | iris-faq.pdf      | df634a   |
| The Iris genus takes its name from the Greek word for rainbow.                                                                   | iris-faq.pdf      | 2aa64c   |
| ...                                                                                                                              | ...               | ...      |
| The generic name Osmanthus is composed of two parts: the Greek words osma meaning smell or fragrance, and anthos meaning flower. | osmanthus-faq.pdf | 850752   |

### [qa_training_data](#training)

The answer here refers to the chunk which contains the information for the question.

| question                                        | answer                                                                  | docname           |
| :---------------------------------------------- |  :--------------------------------------------------------------------- | :---------------- |
| Where does the Iris species take its name from? | The Iris genus takes its name from the Greek word for rainbow.          | iris-faq.pdf      |
| ...                                             | ...                                                                     | ...               |
| Where is osmanthus flavored Pepsi sold?         | PepsiCo makes osmanthus flavored Pepsi for the Chinese domestic market. | osmanthus-faq.pdf |

### [features_df](#inference)

In the case of top-5 chunk retrieval, 

| topn_docname      | topn_scores | topn_rank | query_embedding | chunk_embeddings | hash_key |
| :---------------- | :---------- | :-------- | :-------------- | :--------------- | :------- |
| iris-faq.pdf      | 0.30298     | 0         | [-0.347.., ...] | [0.637.., ...]   | 34ad12   |
| iris-faq.pdf      | 0.33102     | 1         | [-0.347.., ...] | [0.238.., ...]   | 1821aa   |
| ...               | ...         | ...       | ...             | ...              | ...      |
| osmanthus-faq.pdf | 0.36275     | 4         | [-0.347.., ...] | [0.711.., ...]   | d1ua2k   |

## Contribution

We welcome and appreciate all contributions! Please feel free to suggest enhancements, report issues, or submit pull requests.

## License

Copyright ©2024 BlackRock, Inc. or its affiliates. All rights reserved. Distributed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).

## Credits

The implementation of blowfish is a collaborative effort between Javier Makmuri, Thomas R. Barillot, and Alex De Castro.

## Citation

If you found this work helpful, a shout-out in your citations would be very much appreciated! 😊

```
@misc{barillot2024blowfishtopologicalstatisticalsignatures,
      title={Blowfish: Topological and statistical signatures for quantifying ambiguity in semantic search}, 
      author={Thomas Roland Barillot and Alex De Castro},
      year={2024},
      eprint={2406.07990},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2406.07990}, 
}
```