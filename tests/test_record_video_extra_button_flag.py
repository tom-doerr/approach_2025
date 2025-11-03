import sys


def test_add_right_index_inward_flag_appends_label(monkeypatch):
    import record_video as rv
    captured = {"labels": None}

    def fake_record_buttons(labels_csv, cam_index=0):
        captured["labels"] = labels_csv
        # do nothing

    old = sys.argv
    sys.argv = [
        "record_video.py",
        "--buttons", "A,B",
        "--add-button-right-index-inward",
    ]
    try:
        monkeypatch.setattr(rv, "record_buttons", fake_record_buttons)
        rv.main()
    finally:
        sys.argv = old
    assert captured["labels"] is not None
    parts = [s.strip() for s in captured["labels"].split(',') if s.strip()]
    assert "right_index_inward" in parts
    assert "A" in parts and "B" in parts

