from pathlib import Path

from aster_pipeline.artifact_writer import SessionArtifactWriter, atomic_write_bytes


def test_writer_publishes_ordered_atomic_artifacts(tmp_path):
    writer = SessionArtifactWriter()
    first = tmp_path / "annotated" / "field.png"
    second = tmp_path / "result.json"
    writer.submit(lambda: atomic_write_bytes(first, b"annotated"))
    writer.submit(lambda: atomic_write_bytes(second, b'{"status":"completed"}'))
    writer.flush()
    writer.close()

    assert first.read_bytes() == b"annotated"
    assert second.read_text(encoding="utf-8") == '{"status":"completed"}'
    assert not list(tmp_path.rglob("*.tmp"))
