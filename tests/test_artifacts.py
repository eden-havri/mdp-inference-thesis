from __future__ import annotations

from pathlib import Path

from mdp_inference.artifacts import source_tree_sha256


def test_source_tree_hash_is_stable_and_content_sensitive(tmp_path: Path) -> None:
    (tmp_path / "mdp_inference").mkdir()
    source = tmp_path / "mdp_inference" / "module.py"
    source.write_text("value = 1\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    first, first_count = source_tree_sha256(tmp_path)
    second, second_count = source_tree_sha256(tmp_path)
    assert first == second
    assert first_count == second_count == 2
    source.write_text("value = 2\n", encoding="utf-8")
    changed, _ = source_tree_sha256(tmp_path)
    assert changed != first
