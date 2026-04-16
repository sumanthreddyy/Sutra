"""Sutra v0.1–v0.6 validation — imports every module and checks wiring."""

import sys
from pathlib import Path

# Add project root
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
print("SUTRA VALIDATION — v0.1 through v0.6")
print("=" * 60)

# ---- v0.1: Foundation ----
print("\n-- v0.1 Muladhara (Root) --")

check("config.SutraConfig", lambda: __import__("config").SutraConfig)
check("config loads defaults", lambda: __import__("config").SutraConfig.load())

check("providers.base", lambda: __import__("providers.base", fromlist=["Message", "LLMResponse", "ToolDef", "LLMProvider"]))
check("providers.anthropic_provider", lambda: __import__("providers.anthropic_provider", fromlist=["AnthropicProvider"]))
check("providers.openai_provider", lambda: __import__("providers.openai_provider", fromlist=["OpenAIProvider"]))
check("providers.ollama", lambda: __import__("providers.ollama", fromlist=["OllamaProvider"]))

check("memory.types", lambda: __import__("memory.types", fromlist=["MemoryType", "MemoryFile"]))
check("memory.memdir", lambda: __import__("memory.memdir", fromlist=["MemoryDir"]))
check("memory.dream", lambda: __import__("memory.dream", fromlist=["DreamEngine"]))
check("memory.extractor", lambda: __import__("memory.extractor", fromlist=["MemoryExtractor"]))
check("memory.lock", lambda: __import__("memory.lock", fromlist=["ConsolidationLock"]))

check("core.prompt_builder", lambda: __import__("core.prompt_builder", fromlist=["build_system_prompt"]))
check("core.router", lambda: __import__("core.router", fromlist=["Router"]))
check("core.agent", lambda: __import__("core.agent", fromlist=["Agent"]))

check("tools.base", lambda: __import__("tools.base", fromlist=["Tool", "ToolRegistry"]))
check("tools.shell", lambda: __import__("tools.shell", fromlist=["ShellTool"]))
check("tools.file_ops", lambda: __import__("tools.file_ops", fromlist=["FileReadTool", "FileWriteTool", "ListDirTool"]))
check("tools.web_search", lambda: __import__("tools.web_search", fromlist=["WebSearchTool"]))
check("tools.memory_tools", lambda: __import__("tools.memory_tools", fromlist=["MemoryReadTool", "MemoryWriteTool", "MemorySearchTool"]))

check("interfaces.cli", lambda: __import__("interfaces.cli", fromlist=["CLI"]))

# ---- v0.2: Awareness ----
print("\n-- v0.2 Svadhisthana (Sacral) --")

check("memory.vector_store", lambda: __import__("memory.vector_store", fromlist=["VectorStore"]))

# Test VectorStore instantiation (no ChromaDB needed for import)
def test_vector_store_class():
    from memory.vector_store import VectorStore, _recency_score, _frequency_score
    assert callable(_recency_score)
    assert callable(_frequency_score)
    vs = VectorStore.__new__(VectorStore)  # Don't actually init ChromaDB
    assert hasattr(vs, 'search')
    assert hasattr(vs, 'embed_memory')
    assert hasattr(vs, 'reindex_all')
check("VectorStore class structure", test_vector_store_class)

# Test semantic search method exists on MemoryDir
def test_memdir_semantic():
    from memory.memdir import MemoryDir
    assert hasattr(MemoryDir, 'semantic_search')
    assert hasattr(MemoryDir, 'delete_memory')
check("MemoryDir has semantic_search", test_memdir_semantic)

# ---- v0.3: Senses ----
print("\n-- v0.3 Manipura (Solar Plexus) --")

check("senses.voice_input", lambda: __import__("senses.voice_input", fromlist=["VoiceInput"]))
check("senses.voice_output", lambda: __import__("senses.voice_output", fromlist=["VoiceOutput"]))
check("senses.vision", lambda: __import__("senses.vision", fromlist=["Vision"]))
check("senses.screenshot", lambda: __import__("senses.screenshot", fromlist=["ScreenshotTool", "ImageAnalyzeTool"]))

# Test Vision class
def test_vision_class():
    from senses.vision import Vision
    v = Vision(provider="anthropic")
    assert hasattr(v, 'analyze')
    assert hasattr(v, 'analyze_url')
check("Vision class structure", test_vision_class)

# Test VoiceInput/Output classes
def test_voice_classes():
    from senses.voice_input import VoiceInput
    from senses.voice_output import VoiceOutput
    vi = VoiceInput(provider="local")
    vo = VoiceOutput(provider="edge")
    assert hasattr(vi, 'transcribe_file')
    assert hasattr(vi, 'record_and_transcribe')
    assert hasattr(vo, 'speak')
    assert hasattr(vo, 'speak_and_play')
check("Voice I/O class structure", test_voice_classes)

# ---- v0.4: Coordination ----
print("\n-- v0.4 Anahata (Heart) --")

check("coordination.shared_context", lambda: __import__("coordination.shared_context", fromlist=["SharedContext", "SubTask", "CoordinationPlan", "TaskStatus"]))
check("coordination.decomposer", lambda: __import__("coordination.decomposer", fromlist=["decompose_task"]))
check("coordination.worker", lambda: __import__("coordination.worker", fromlist=["Worker"]))
check("coordination.coordinator", lambda: __import__("coordination.coordinator", fromlist=["Coordinator"]))

# Test task status enum
def test_task_types():
    from coordination.shared_context import TaskStatus, SubTask, CoordinationPlan
    assert TaskStatus.PENDING.value == "pending"
    assert TaskStatus.COMPLETED.value == "completed"
    t = SubTask(id="t1", description="test")
    assert t.status == TaskStatus.PENDING
    plan = CoordinationPlan(original_task="test", subtasks=[t])
    assert not plan.is_complete
    ready = plan.get_ready_tasks()
    assert len(ready) == 1
check("Task types + plan logic", test_task_types)

# Test SharedContext
def test_shared_context():
    import asyncio
    from coordination.shared_context import SharedContext
    ctx = SharedContext()
    async def run():
        await ctx.set("key", "value")
        v = await ctx.get("key")
        assert v == "value"
        await ctx.append("list", 1)
        await ctx.append("list", 2)
        l = await ctx.get("list")
        assert l == [1, 2]
    asyncio.run(run())
check("SharedContext async operations", test_shared_context)

# Test Coordinator instantiation
def test_coordinator_init():
    from coordination.coordinator import Coordinator
    from tools.base import ToolRegistry
    # Just verify it can be constructed (no real router)
    c = Coordinator.__new__(Coordinator)
    assert hasattr(c, 'coordinate')
    assert hasattr(c, 'get_status')
check("Coordinator class structure", test_coordinator_init)

# ---- v0.5: Autonomy ----
print("\n-- v0.5 Vishuddha (Throat) --")

check("autonomy.scheduler", lambda: __import__("autonomy.scheduler", fromlist=["Scheduler", "ScheduledTask", "ScheduleType"]))
check("autonomy.watchers", lambda: __import__("autonomy.watchers", fromlist=["FileWatcher", "IdleDetector"]))
check("autonomy.proactive", lambda: __import__("autonomy.proactive", fromlist=["ProactiveEngine"]))
check("autonomy.goals", lambda: __import__("autonomy.goals", fromlist=["GoalTracker", "Goal"]))

def test_scheduler_types():
    from autonomy.scheduler import ScheduledTask, ScheduleType
    import asyncio
    async def noop(): pass
    t = ScheduledTask(
        name="test",
        description="test task",
        schedule_type=ScheduleType.INTERVAL,
        handler=noop,
        interval_seconds=60,
    )
    assert t.schedule_type == ScheduleType.INTERVAL
    assert t.interval_seconds == 60
check("Scheduler task types", test_scheduler_types)

def test_idle_detector():
    from autonomy.watchers import IdleDetector
    d = IdleDetector()
    d.touch()
    assert d.idle_seconds < 1.0  # just touched, should be near zero
check("IdleDetector touch/idle_seconds", test_idle_detector)

def test_goal_tracker():
    import tempfile, shutil
    from autonomy.goals import GoalTracker, Goal
    tmpdir = tempfile.mkdtemp()
    from memory.memdir import MemoryDir
    md = MemoryDir(tmpdir)
    gt = GoalTracker(md)
    g = gt.create_goal("Test Goal")
    assert g.title == "Test Goal"
    assert g.progress == 0
    goals = gt.list_goals()
    assert len(goals) >= 1
    updated = gt.update_progress(g.id, 50)
    assert updated.progress == 50
    shutil.rmtree(tmpdir, ignore_errors=True)
check("GoalTracker CRUD", test_goal_tracker)

# ---- v0.6: Intuition ----
print("\n-- v0.6 Ajna (Third Eye) --")

check("intuition.patterns", lambda: __import__("intuition.patterns", fromlist=["PatternDetector"]))
check("intuition.predictions", lambda: __import__("intuition.predictions", fromlist=["PredictionEngine"]))

def test_pattern_detector():
    import tempfile, shutil
    from intuition.patterns import PatternDetector
    tmpdir = tempfile.mkdtemp()
    pd = PatternDetector(transcript_dir=tmpdir)
    summary = pd.get_summary(days=7)
    assert isinstance(summary, str)
    shutil.rmtree(tmpdir, ignore_errors=True)
check("PatternDetector summary", test_pattern_detector)

def test_prediction_engine():
    import tempfile, shutil
    from intuition.predictions import PredictionEngine
    from intuition.patterns import PatternDetector
    from autonomy.goals import GoalTracker
    from memory.memdir import MemoryDir
    tmpdir = tempfile.mkdtemp()
    pd = PatternDetector(transcript_dir=tmpdir)
    md = MemoryDir(tmpdir)
    gt = GoalTracker(md)
    pe = PredictionEngine(pattern_detector=pd, goal_tracker=gt)
    assert hasattr(pe, 'predict')
    assert hasattr(pe, '_prediction_cache')
    shutil.rmtree(tmpdir, ignore_errors=True)
check("PredictionEngine structure", test_prediction_engine)

# ---- Cross-cutting ----
print("\n-- Cross-cutting checks --")

# Tool registry wiring
def test_tool_registry():
    from tools.base import ToolRegistry
    from tools.shell import ShellTool
    from tools.file_ops import FileReadTool
    from tools.web_search import WebSearchTool
    from tools.memory_tools import MemoryReadTool
    from memory.memdir import MemoryDir
    import tempfile, os

    reg = ToolRegistry()
    reg.register(ShellTool())
    reg.register(FileReadTool())
    reg.register(WebSearchTool())

    tmpdir = tempfile.mkdtemp()
    md = MemoryDir(tmpdir)
    reg.register(MemoryReadTool(md))

    assert len(reg.list_names()) == 4
    defs = reg.get_tool_defs()
    assert len(defs) == 4
    assert all(d.name for d in defs)

    # Cleanup
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)
check("ToolRegistry with 4 tools", test_tool_registry)

# MemoryFile roundtrip
def test_memory_roundtrip():
    from memory.types import MemoryFile, MemoryType
    m = MemoryFile(
        path="test.md",
        name="Test",
        description="Test memory",
        type=MemoryType.USER,
        content="Hello world",
    )
    frontmatter = m.to_frontmatter()
    assert "name:" in frontmatter and "Test" in frontmatter
    assert "type: user" in frontmatter
    # Parse back
    parsed = MemoryFile.from_file("test.md", frontmatter)
    assert parsed.name == "Test"
    assert parsed.type == MemoryType.USER
    assert parsed.content == "Hello world"
check("Memory frontmatter roundtrip", test_memory_roundtrip)

# PromptBuilder
def test_prompt_builder():
    import tempfile
    from core.prompt_builder import build_system_prompt
    from memory.memdir import MemoryDir
    tmpdir = tempfile.mkdtemp()
    md = MemoryDir(tmpdir)
    prompt = build_system_prompt(md, user_name="TestUser")
    assert "Sutra" in prompt
    assert "TestUser" in prompt
    assert "MEMORY.md" in prompt
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)
check("Prompt builder generates valid prompt", test_prompt_builder)

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
