import numpy as np
import pandas as pd
import shap
from sklearn.neighbors import KernelDensity

from blowfish.calculations import calculate_relevant_features
from blowfish.inference import AmbiguityScorer, FeedbackDecider


def test_scorer_probability_remains_finite_far_from_training_samples():
    kde = KernelDensity(bandwidth=0.1).fit([[0.0, 0.0], [1.0, 0.0]])
    scorer = AmbiguityScorer(kde, pd.DataFrame(columns=[
        'docname', 'chunk_embedding', 'label', 'silhouette_score', 'hash_key']))
    np.testing.assert_allclose(scorer.get_correctness_probability([100.0]), [0.5])


def test_decider_probability_remains_finite_far_from_training_samples():
    decider = FeedbackDecider.__new__(FeedbackDecider)
    decider.KDE = KernelDensity(bandwidth=0.1).fit([[0.0, 0.0], [1.0, 0.0]])
    np.testing.assert_allclose(decider.KDE_prediction(np.array([[100.0]])), [0.5])


def test_document_spread_produces_document_feedback():
    decider = FeedbackDecider.__new__(FeedbackDecider)
    decider.KDE_features_order = ['top_k_doc_spread', 'top_k_topic_spread']
    decider.explainer = lambda values: shap.Explanation(
        values=np.array([[-0.8, -0.01]]), data=values)
    assert decider.explain_query(pd.DataFrame({
        'top_k_doc_spread': [0.5], 'top_k_topic_spread': [0.2]})) == 'docspread'


def test_requested_feature_does_not_require_unrelated_columns():
    features = calculate_relevant_features(
        pd.DataFrame({'docname': ['a', 'b', 'a']}), ['top_k_doc_spread'])
    assert features == {'top_k_doc_spread': 2 / 3}


def test_high_accuracy_sampling_scores_the_binary_positive_candidate():
    class Density:
        def sample(self, n_samples):
            return np.array([[0.6, 2.0]])

        def score(self, sample):
            return 0.0 if sample[0, 0] == 1.0 else -2.0

    decider = FeedbackDecider.__new__(FeedbackDecider)
    decider.KDE = Density()
    # A sampled class coordinate is noisy, but conditioning uses class 0/1.
    # Abort promptly on the old implementation instead of allowing an infinite loop.
    original_sample = decider.KDE.sample
    calls = 0
    def sample(n_samples):
        nonlocal calls
        calls += 1
        assert calls < 3, 'did not accept the binary positive candidate'
        return original_sample(n_samples)
    decider.KDE.sample = sample
    np.testing.assert_array_equal(decider.get_high_accuracy_samples(1), [[2.0]])
