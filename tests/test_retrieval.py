from evidencerag.retrieval import combined_score, lexical_overlap


def test_lexical_overlap_rewards_matching_terms() -> None:
    matching = lexical_overlap("critical incident SLA", "Critical incident SLA is fifteen minutes")
    unrelated = lexical_overlap("critical incident SLA", "Vacation policy and office hours")

    assert matching > unrelated


def test_combined_score_uses_vector_and_lexical_signal() -> None:
    relevant = combined_score("rollback release", "rollback the release", 0.1)
    weak = combined_score("rollback release", "unrelated content", 0.8)

    assert relevant > weak
