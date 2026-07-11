import pytest

from quant_agent.tools.github_tool import _parse_owner_repo


def test_parse_owner_repo_accepts_plain_github_url():
    assert _parse_owner_repo("https://github.com/org/repo") == ("org", "repo")


@pytest.mark.parametrize("url", [
    "https://github.com/org/repo/../../user",
    "https://evil.example/org/repo",
    "https://github.com/org/repo?ref=x",
])
def test_parse_owner_repo_rejects_noncanonical_paths(url):
    with pytest.raises(ValueError):
        _parse_owner_repo(url)
