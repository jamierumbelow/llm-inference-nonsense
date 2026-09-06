import pytest

from harness.profiles import PROFILES, get_profile


def test_1b_and_8b_exist() -> None:
    assert get_profile("1b").repo_id.startswith("meta-llama/Llama-3.2-1B")
    assert get_profile("8b").repo_id.startswith("meta-llama/Llama-3.1-8B")


def test_repo_ids_unique() -> None:
    repo_ids = [p.repo_id for p in PROFILES.values()]
    assert len(repo_ids) == len(set(repo_ids))


def test_unknown_profile_lists_known() -> None:
    with pytest.raises(KeyError, match="1b, 8b"):
        get_profile("bogus")
