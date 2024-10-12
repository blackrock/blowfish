from blowfish.inference.scorer import AmbiguityScorer
import pandas as pd
from pandas.testing import assert_frame_equal

topic_features = {
    "docname": ["doc1", "doc2"],
    "chunk_embedding": [1.512, 2.123],
    "label": [1, 3],
    "silhouette_score": [0.719, 0.890],
    "hash_key": ["hash_key_1", "hash_key_2"]
}
topics_df = pd.DataFrame(topic_features)
scorer = AmbiguityScorer(kde=None, topics_df=topics_df)

def test_join_topics_to_query_and_chunks_topics():
    features = {
        "topn_docname": ["doc1", "doc2"],
        "topn_scores": [0.33, 0.21],
        "topn_rank": [0, 1],
        "query_embedding": [1.321, 2.221],
        "chunk_embeddings": [1.512, 2.123],
        "hash_key": ["hash_key_1", "hash_key_2"]
    }

    expected_values = {
        "docname": ["doc1", "doc2"],
        "score": [0.33, 0.21],
        "rank": [0, 1],
        "query_embedding": [1.321, 2.221],
        "chunk_embeddings": [1.512, 2.123],
        "hash_key": ["hash_key_1", "hash_key_2"],
        "label": [1, 3],
        "silhouette_score": [0.719, 0.890],
        "topic_label": ["doc1_1", "doc2_3"]
    }
    
    expected = pd.DataFrame(expected_values)

    test_df = pd.DataFrame(features)
    actual = scorer.join_topics_to_query_and_chunks(test_df).drop('index', axis=1)

    assert_frame_equal(actual, expected)


def test_join_topics_to_query_and_chunks_missing_columns():
    bad_features = {
        "topn_docname": ["doc1", "doc2"],
        "topn_scores": [0.33, 0.21],
        "topn_rank": [0, 1],
        "query_embedding": [1.321, 2.221],
        "chunk_embeddings": [1.512, 2.123]
    }
    fail_df = pd.DataFrame(bad_features)
    res = scorer.join_topics_to_query_and_chunks(fail_df)

    assert res == None


def test_format_topics():
    topic_features = {
        "docname": ["doc1", "doc2"],
        "chunk_embedding": [1.512, 2.123],
        "label": [1, 3],
        "silhouette_score": [0.719, 0.890],
        "hash_key": ["hash_key_1", "hash_key_2"]
    }

    expected_features = {
        "docname": ["doc1", "doc2"],
        "chunk_embedding": [1.512, 2.123],
        "label": [1, 3],
        "silhouette_score": [0.719, 0.890],
        "hash_key": ["hash_key_1", "hash_key_2"],
        "cluster_id": ["doc1_1", "doc2_3"]
    }

    topics_df = pd.DataFrame(topic_features)
    actual = scorer.format_topics(topics_df)
    expected = pd.DataFrame(expected_features).set_index(["docname"])
    
    assert_frame_equal(actual, expected)


def test_format_topics_missing_columns():
    bad_features = {
        "docname": ["doc1", "doc2"],
        "chunk_embedding": [1.512, 2.123],
        "label": [1, 3],
        "silhouette_score": [0.719, 0.890]
    }
    fail_df = pd.DataFrame(bad_features)
    res = scorer.format_topics(fail_df)

    assert res == None