"""Unit tests for corpus acquisition."""

from __future__ import annotations

import pytest

from llm_lab.data.corpus import load_documents


def test_load_documents_unknown_corpus_raises() -> None:
    with pytest.raises(ValueError):
        load_documents("does_not_exist")
