from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train import reset_ttt_on_doc_switch


class DummyModel:
    def __init__(self):
        self.reset_calls = 0

    def reset_ttt_state(self):
        self.reset_calls += 1


def test_reset_ttt_on_doc_switch_detects_new_documents():
    model = DummyModel()
    last_doc = None

    # First batch introduces docA twice and docB once
    last_doc = reset_ttt_on_doc_switch(model, ["docA", "docA", "docB"], last_doc)
    assert model.reset_calls == 2
    assert last_doc == "docB"

    # Second batch contains docB again and a None entry -> no extra reset
    last_doc = reset_ttt_on_doc_switch(model, [None, "docB"], last_doc)
    assert model.reset_calls == 2
    assert last_doc == "docB"

    # Third batch brings a brand new document
    last_doc = reset_ttt_on_doc_switch(model, ["docC"], last_doc)
    assert model.reset_calls == 3
    assert last_doc == "docC"
