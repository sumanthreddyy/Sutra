"""Tests for v0.7 non-lossy memory system.

Validates:
1. Archive instead of delete — files survive, flagged as archived
2. Transcript chunking — raw conversations split into searchable chunks
3. Multi-layer search — memories + transcripts searched together
4. Memory frontmatter — archived flag and source roundtrip
5. MEMORY_ARCHIVE.md — overflow entries preserved
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

errors = []
passed = []


def check(label, fn):
    try:
        fn()
        passed.append(label)
        print(f"  [OK] {label}")
    except Exception as e:
        errors.append((label, str(e)))
        print(f"  [FAIL] {label}: {e}")


print("=" * 60)
print("SUTRA v0.7 — Non-Lossy Memory Tests")
print("=" * 60)


# ---- 1. Archive instead of delete ----
print("\n-- Archive instead of delete --")


def test_archive_preserves_file():
    tmpdir = tempfile.mkdtemp()
    from memory.memdir import MemoryDir
    from memory.types import MemoryFile, MemoryType

    md = MemoryDir(tmpdir)
    mem = MemoryFile(
        path="test_fact.md", name="Test Fact",
        description="A fact that should never be lost",
        type=MemoryType.USER, content="The sky is blue.",
    )
    md.save_memory(mem)

    # File exists on disk
    assert (Path(tmpdir) / "test_fact.md").exists()

    # "Delete" (which now archives)
    result = md.delete_memory("test_fact.md")
    assert result is True

    # File STILL exists on disk — not deleted
    assert (Path(tmpdir) / "test_fact.md").exists(), "File was deleted! Should be archived."

    # File is now marked as archived
    reread = md.read_memory("test_fact.md")
    assert reread is not None
    assert reread.archived is True, f"Expected archived=True, got {reread.archived}"

    # Content is preserved
    assert reread.content == "The sky is blue."

    shutil.rmtree(tmpdir, ignore_errors=True)
check("Archive preserves file on disk", test_archive_preserves_file)


def test_archive_moves_to_archive_index():
    tmpdir = tempfile.mkdtemp()
    from memory.memdir import MemoryDir
    from memory.types import MemoryFile, MemoryType

    md = MemoryDir(tmpdir)
    mem = MemoryFile(
        path="old_fact.md", name="Old Fact",
        description="This will be archived",
        type=MemoryType.PROJECT, content="We used MySQL before.",
    )
    md.save_memory(mem)

    # Active index has the entry
    active_index = md.read_index()
    assert "old_fact.md" in active_index

    # Archive it
    md.archive_memory("old_fact.md")

    # Active index should NOT have it anymore
    active_index = md.read_index()
    assert "old_fact.md" not in active_index

    # Archive index SHOULD have it
    archive_path = Path(tmpdir) / "MEMORY_ARCHIVE.md"
    assert archive_path.exists(), "MEMORY_ARCHIVE.md was not created"
    archive_content = archive_path.read_text(encoding="utf-8")
    assert "old_fact.md" in archive_content

    shutil.rmtree(tmpdir, ignore_errors=True)
check("Archive entry moves to MEMORY_ARCHIVE.md", test_archive_moves_to_archive_index)


def test_list_memories_excludes_archived():
    tmpdir = tempfile.mkdtemp()
    from memory.memdir import MemoryDir
    from memory.types import MemoryFile, MemoryType

    md = MemoryDir(tmpdir)
    # Save two memories
    md.save_memory(MemoryFile(
        path="active.md", name="Active",
        description="Still relevant", type=MemoryType.USER, content="Active content",
    ))
    md.save_memory(MemoryFile(
        path="old.md", name="Old",
        description="Outdated", type=MemoryType.USER, content="Old content",
    ))

    # Archive one
    md.archive_memory("old.md")

    # list_memories (default) should only show active
    active = md.list_active_memories()
    assert len(active) == 1
    assert active[0].name == "Active"

    # list_memories with include_archived should show both
    all_mems = md.list_memories(include_archived=True)
    assert len(all_mems) == 2

    # list_archived should show only archived
    archived = md.list_archived_memories()
    assert len(archived) == 1
    assert archived[0].name == "Old"
    assert archived[0].archived is True

    shutil.rmtree(tmpdir, ignore_errors=True)
check("list_memories respects archived flag", test_list_memories_excludes_archived)


# ---- 2. Transcript chunking ----
print("\n-- Transcript Store --")


def test_transcript_chunking():
    tmpdir = tempfile.mkdtemp()
    from memory.transcript_store import TranscriptStore

    ts = TranscriptStore(tmpdir)

    # Create a fake transcript
    transcript_path = Path(tmpdir) / "session_001.jsonl"
    turns = [
        {"role": "user", "content": "What's the best database for this?"},
        {"role": "assistant", "content": "Depends on your use case. For concurrent writes, Postgres."},
        {"role": "user", "content": "We'll have about 10GB of data."},
        {"role": "assistant", "content": "Postgres handles that well. SQLite would struggle."},
        {"role": "user", "content": "Let's go with Postgres then."},
        {"role": "assistant", "content": "Good choice. I'll remember this decision."},
        {"role": "user", "content": "Also, the API key rotates every 90 days."},
        {"role": "assistant", "content": "Noted. I'll track that."},
        {"role": "user", "content": "Thanks!"},
        {"role": "assistant", "content": "You're welcome!"},
    ]
    with open(transcript_path, "w") as f:
        for turn in turns:
            f.write(json.dumps(turn) + "\n")

    # Chunk it
    chunks = ts.chunk_transcript(transcript_path)
    assert len(chunks) > 0, "No chunks produced"

    # Each chunk should have turns and flattened text
    for chunk in chunks:
        assert chunk.id.startswith("session_001:")
        assert len(chunk.turns) > 0
        assert len(chunk.text) > 0
        assert "[user]:" in chunk.text or "[assistant]:" in chunk.text

    # Chunks should overlap (50% overlap with CHUNK_SIZE=6, step=3)
    if len(turns) > 6:
        assert len(chunks) >= 2, "Expected overlapping chunks"

    shutil.rmtree(tmpdir, ignore_errors=True)
check("Transcript chunking with overlap", test_transcript_chunking)


def test_transcript_chunk_content():
    tmpdir = tempfile.mkdtemp()
    from memory.transcript_store import TranscriptStore

    ts = TranscriptStore(tmpdir)

    transcript_path = Path(tmpdir) / "api_key_chat.jsonl"
    turns = [
        {"role": "user", "content": "The API key rotates every 90 days, don't forget."},
        {"role": "assistant", "content": "Got it, 90-day rotation cycle for the API key."},
    ]
    with open(transcript_path, "w") as f:
        for t in turns:
            f.write(json.dumps(t) + "\n")

    chunks = ts.chunk_transcript(transcript_path)
    assert len(chunks) >= 1

    # The important detail should be in the chunk text
    combined_text = " ".join(c.text for c in chunks)
    assert "90 days" in combined_text, "Important detail lost in chunking!"
    assert "API key" in combined_text, "Important detail lost in chunking!"

    shutil.rmtree(tmpdir, ignore_errors=True)
check("Transcript preserves important details", test_transcript_chunk_content)


def test_transcript_store_stats():
    tmpdir = tempfile.mkdtemp()
    from memory.transcript_store import TranscriptStore

    ts = TranscriptStore(tmpdir)

    # Create 2 transcripts
    for name in ["s1.jsonl", "s2.jsonl"]:
        p = Path(tmpdir) / name
        with open(p, "w") as f:
            f.write(json.dumps({"role": "user", "content": "hello"}) + "\n")
            f.write(json.dumps({"role": "assistant", "content": "hi"}) + "\n")

    stats = ts.get_stats()
    assert stats["transcript_files"] == 2
    assert stats["total_turns"] == 4

    shutil.rmtree(tmpdir, ignore_errors=True)
check("Transcript store stats", test_transcript_store_stats)


# ---- 3. Memory frontmatter roundtrip ----
print("\n-- Frontmatter with new fields --")


def test_archived_frontmatter_roundtrip():
    from memory.types import MemoryFile, MemoryType

    mem = MemoryFile(
        path="archived_test.md", name="Archived Test",
        description="Was relevant, now archived",
        type=MemoryType.PROJECT, content="Old project data",
        archived=True, source="consolidated",
    )
    raw = mem.to_frontmatter()
    assert "archived: true" in raw
    assert "source: consolidated" in raw

    # Parse back
    parsed = MemoryFile.from_file("archived_test.md", raw)
    assert parsed.archived is True
    assert parsed.source == "consolidated"
    assert parsed.content == "Old project data"
    assert parsed.type == MemoryType.PROJECT
check("Archived + source frontmatter roundtrip", test_archived_frontmatter_roundtrip)


def test_default_fields_not_in_frontmatter():
    from memory.types import MemoryFile, MemoryType

    mem = MemoryFile(
        path="normal.md", name="Normal",
        description="Regular memory", type=MemoryType.USER,
        content="hello",
    )
    raw = mem.to_frontmatter()
    # Default values should NOT be written (clean frontmatter)
    assert "archived" not in raw
    assert "source" not in raw
check("Default fields omitted from frontmatter", test_default_fields_not_in_frontmatter)


# ---- 4. Unified search ----
print("\n-- Unified multi-layer search --")


def test_unified_search_without_vector_store():
    tmpdir = tempfile.mkdtemp()
    from memory.memdir import MemoryDir
    from memory.types import MemoryFile, MemoryType

    md = MemoryDir(tmpdir)
    md.save_memory(MemoryFile(
        path="postgres.md", name="DB Choice",
        description="Chose Postgres for concurrent writes",
        type=MemoryType.PROJECT, content="We chose Postgres because of concurrent writes.",
    ))

    # unified_search without vector store falls back to keyword
    results = md.unified_search("Postgres", n_results=5)
    assert len(results) >= 1
    assert any("Postgres" in r.get("content", "") for r in results)

    shutil.rmtree(tmpdir, ignore_errors=True)
check("Unified search (keyword fallback)", test_unified_search_without_vector_store)


def test_unified_search_has_source_tag():
    tmpdir = tempfile.mkdtemp()
    from memory.memdir import MemoryDir
    from memory.types import MemoryFile, MemoryType

    md = MemoryDir(tmpdir)
    md.save_memory(MemoryFile(
        path="fact.md", name="Fact",
        description="A test fact", type=MemoryType.USER,
        content="Testing source tags",
    ))

    results = md.unified_search("test fact", n_results=5)
    for r in results:
        assert "source" in r, f"Result missing source tag: {r}"
    
    shutil.rmtree(tmpdir, ignore_errors=True)
check("Unified search results have source tag", test_unified_search_has_source_tag)


# ---- 5. Non-lossy guarantee ----
print("\n-- Non-lossy guarantee --")


def test_full_lifecycle_nothing_lost():
    """Simulate the full lifecycle: create → extract → dream → verify nothing lost."""
    tmpdir = tempfile.mkdtemp()
    from memory.memdir import MemoryDir
    from memory.types import MemoryFile, MemoryType

    md = MemoryDir(tmpdir)

    # Simulate extraction: save 3 memories from different conversations
    mem1 = MemoryFile(
        path="auth_decision.md", name="Auth Decision",
        description="Chose Clerk over Auth0",
        type=MemoryType.PROJECT, content="Kai recommended Clerk. Pricing + DX.",
    )
    mem2 = MemoryFile(
        path="db_choice.md", name="DB Choice",
        description="Postgres for concurrent writes",
        type=MemoryType.PROJECT, content="Postgres because 10GB+ data.",
    )
    mem3 = MemoryFile(
        path="api_rotation.md", name="API Key Rotation",
        description="Keys rotate every 90 days",
        type=MemoryType.REFERENCE, content="API key rotates every 90 days.",
    )

    for m in [mem1, mem2, mem3]:
        md.save_memory(m)

    # Verify all 3 are active
    assert len(md.list_active_memories()) == 3

    # Simulate dream consolidation: archive old memories, create consolidated
    md.archive_memory("auth_decision.md")
    md.archive_memory("db_choice.md")

    # Create a consolidated memory that merges them
    consolidated = MemoryFile(
        path="infra_decisions.md", name="Infrastructure Decisions",
        description="Consolidated auth + db decisions",
        type=MemoryType.PROJECT,
        content="Auth: Clerk (recommended by Kai, pricing + DX). DB: Postgres (concurrent writes, 10GB+).",
        source="consolidated",
    )
    md.save_memory(consolidated)

    # Verify: 2 active (api_rotation + consolidated), 2 archived
    active = md.list_active_memories()
    assert len(active) == 2, f"Expected 2 active, got {len(active)}: {[m.name for m in active]}"

    archived = md.list_archived_memories()
    assert len(archived) == 2, f"Expected 2 archived, got {len(archived)}"

    # CRITICAL: nothing was lost. All 4 files exist on disk.
    all_files = list(Path(tmpdir).glob("*.md"))
    md_names = {f.name for f in all_files}
    assert "auth_decision.md" in md_names, "Original was deleted!"
    assert "db_choice.md" in md_names, "Original was deleted!"
    assert "api_rotation.md" in md_names
    assert "infra_decisions.md" in md_names

    # Archived files still have their content
    auth = md.read_memory("auth_decision.md")
    assert "Clerk" in auth.content, "Archived content was corrupted!"
    assert auth.archived is True

    # MEMORY_ARCHIVE.md exists with the archived entries
    archive_idx = (Path(tmpdir) / "MEMORY_ARCHIVE.md").read_text(encoding="utf-8")
    assert "auth_decision.md" in archive_idx
    assert "db_choice.md" in archive_idx

    shutil.rmtree(tmpdir, ignore_errors=True)
check("Full lifecycle — nothing lost", test_full_lifecycle_nothing_lost)


# ---- Summary ----
print("\n" + "=" * 60)
print(f"PASSED: {len(passed)}")
print(f"FAILED: {len(errors)}")
if errors:
    print("\nFailures:")
    for label, err in errors:
        print(f"  - {label}: {err}")
print("=" * 60)

sys.exit(1 if errors else 0)
