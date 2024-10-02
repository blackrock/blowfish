from blowfish.training.disambiguator_training import DisambiguationModelGenerator
import pandas as pd
from pandas.testing import assert_frame_equal

model = DisambiguationModelGenerator()

def test_format_qa_eval_df():
    features = {
        "query_id": [0],
        "chunk_topn_docname": [["doc1", "doc2", "doc1"]],
        "chunk_topn_scores": [[0.32, 0.35, 0.36]],
        "topic_labels": [[72, 27, 13]],
        "doc_match": [[1, 0, 1]],
        "chunk_match": [[0, 0, 1]],
        "query_embedding": [[0.11, 0.22]],
        "chunk_embeddings": [[[0.12, 0.13], [0.22, 0.23], [0.32, 0.33]]],
        "silhouette_score": [[0.76, 0.52, 0.35]]
    }

    expected_features = {
        "query_id": [0, 0, 0],
        "docname": ["doc1", "doc2", "doc1"],
        "score": [0.32, 0.35, 0.36],
        "label": [72, 27, 13],
        "doc_match": [1, 0, 1],
        "chunk_match": [0, 0, 1],
        "query_embedding": [[0.11, 0.22], [0.11, 0.22], [0.11, 0.22]],
        "chunk_embeddings": [[0.12, 0.13], [0.22, 0.23], [0.32, 0.33]],
        "silhouette_score": [0.76, 0.52, 0.35],
        "rank": [0, 1, 2],
        "topic_label": ["doc1_72", "doc2_27", "doc1_13"]
    }


    expected = pd.DataFrame(expected_features)
    qa_eval_df = pd.DataFrame(features)
    actual = model.format_qa_eval_df(qa_eval_df)

    """
        These type conversions are needed because pd.DataFrame.explode() keeps the columns as type object due to it being exploded from a list.
        This type doesn't affect training downsteam, it just affects the frame assertion function
    """
    actual["score"] = actual["score"].astype(float)
    actual["label"] = actual["label"].astype(int)
    actual["doc_match"] = actual["doc_match"].astype(int)
    actual["chunk_match"] = actual["chunk_match"].astype(int)
    actual["silhouette_score"] = actual["silhouette_score"].astype(float)
    actual["rank"] = actual["rank"].astype(int)

    assert_frame_equal(actual, expected)


def test_generate_queries_features_filter_sub_df_with_one_chunk():
    features = {
        "query_id": [0, 0, 1, 1],
        "docname": ["doc1", "doc2", "doc1", "doc1"],
        "score": [0.32, 0.35, 0.36, 0.36],
        "label": [72, 27, 13, 13],
        "doc_match": [1, 0, 1, 1],
        "chunk_match": [0, 0, 1, 1],
        "query_embedding": [[0.11, 0.22], [0.11, 0.22], [0.12, 0.23], [0.12, 0.23]],
        "chunk_embeddings": [[0.12, 0.13], [0.22, 0.23], [0.32, 0.33], [0.32, 0.33]],
        "silhouette_score": [0.76, 0.52, 0.35, 0.35],
        "rank": [0, 1, 0, 0],
        "topic_label": ["doc1_72", "doc2_27", "doc1_13", "doc1_13"]
    }

    df = pd.DataFrame(features)
    actual = model.generate_queries_features(df)
    assert len(actual) == 1