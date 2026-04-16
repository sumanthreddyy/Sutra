"""Main agent loop — message → think → act → respond.

The core orchestration that ties everything together:
- Receives user messages
- Builds dynamic system prompt with current memory
- Routes to appropriate LLM provider
- Handles tool calls
- Triggers background memory extraction
- Triggers dream consolidation when gates pass
- Saves conversation transcripts
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from config import SutraConfig
from core.prompt_builder import build_system_prompt
from core.router import Router
from memory.dream import DreamEngine
from memory.extractor import MemoryExtractor
from memory.memdir import MemoryDir
from providers.base import LLMResponse, Message
from tools.base import ToolRegistry
from tools.file_ops import FileReadTool, FileWriteTool, ListDirTool
from tools.memory_tools import MemoryReadTool, MemorySearchTool, MemoryWriteTool
from tools.shell import ShellTool
from tools.web_search import WebSearchTool

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 10  # Prevent infinite tool-call loops


class Agent:
    """Main Sutra agent — the brain that orchestrates everything."""

    def __init__(self, config: SutraConfig, router: Router, vector_store: Any = None, vision: Any = None, voice_in: Any = None, voice_out: Any = None):
        self.config = config
        self.router = router

        # Memory system
        self.memory_dir = MemoryDir(config.memory.base_dir, vector_store=vector_store)
        self.vector_store = vector_store

        # Senses (v0.3)
        self.vision = vision
        self.voice_in = voice_in
        self.voice_out = voice_out

        # Dream engine
        self.dream = DreamEngine(
            memory_dir=self.memory_dir,
            transcript_dir=config.conversations.transcript_dir,
            min_hours=config.memory.dream.min_hours,
            min_sessions=config.memory.dream.min_sessions,
        )

        # Memory extractor
        self.extractor = MemoryExtractor(
            memory_dir=self.memory_dir,
            min_new_messages=config.memory.extract.min_new_messages,
        )

        # Tool registry
        self.tools = ToolRegistry()
        self.tools.register(ShellTool())
        self.tools.register(FileReadTool())
        self.tools.register(FileWriteTool())
        self.tools.register(ListDirTool())
        self.tools.register(MemoryReadTool(self.memory_dir))
        self.tools.register(MemoryWriteTool(self.memory_dir))
        self.tools.register(MemorySearchTool(self.memory_dir))
        self.tools.register(WebSearchTool())

        # Vision tools (v0.3)
        if self.vision:
            from senses.screenshot import ScreenshotTool, ImageAnalyzeTool
            self.tools.register(ScreenshotTool(vision=self.vision))
            self.tools.register(ImageAnalyzeTool(vision=self.vision))

        # Coordinator (v0.4)
        from coordination.coordinator import Coordinator
        self.coordinator = Coordinator(router=self.router, tools=self.tools)

        # Autonomy (v0.5)
        from autonomy.scheduler import Scheduler
        from autonomy.proactive import ProactiveEngine
        from autonomy.goals import GoalTracker
        from autonomy.watchers import IdleDetector
        self.scheduler = Scheduler()
        self.proactive = ProactiveEngine(
            transcript_dir=config.conversations.transcript_dir,
            memory_dir=self.memory_dir,
        )
        self.goals = GoalTracker(self.memory_dir)
        self.idle_detector = IdleDetector()

        # Intuition (v0.6)
        from intuition.patterns import PatternDetector
        from intuition.predictions import PredictionEngine
        self.pattern_detector = PatternDetector(config.conversations.transcript_dir)
        self.predictions = PredictionEngine(self.pattern_detector, self.goals)

        # Conversation state
        self.messages: list[Message] = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.turn_count = 0
        self.total_usage = {"input_tokens": 0, "output_tokens": 0}

        # Ensure transcript directory exists
        Path(config.conversations.transcript_dir).mkdir(parents=True, exist_ok=True)

    def _build_system_prompt(self, user_input: str = "") -> str:
        """Build the current system prompt with fresh memory."""
        user_name = getattr(self.config, "user_name", None) or "friend"
        user_mem = self.memory_dir.read_memory("user_name.md")
        if user_mem:
            user_name = user_mem.content.strip()

        # Inject active goals as extra context
        goals_ctx = self.goals.get_active_summary()

        return build_system_prompt(
            memory_dir=self.memory_dir,
            user_name=user_name,
            relevant_query=user_input,
            extra_context=goals_ctx,
        )

    async def chat(self, user_input: str) -> str:
        """Process a user message and return the response."""
        self.turn_count += 1
        self.idle_detector.touch()

        # Add user message
        self.messages.append(Message(role="user", content=user_input))

        # Build system prompt with current memory state
        system_prompt = self._build_system_prompt(user_input)

        # Build messages for LLM (system + recent conversation)
        max_context = self.config.conversations.max_context_messages
        recent = self.messages[-max_context:]
        llm_messages = [Message(role="system", content=system_prompt)] + recent

        # Get tool definitions
        tool_defs = self.tools.get_tool_defs()

        # Tool-call loop — keep going until the LLM produces a final text response
        response: LLMResponse | None = None
        for _ in range(MAX_TOOL_ROUNDS):
            response = await self.router.complete(
                messages=llm_messages,
                task_type="default",
                tools=tool_defs if tool_defs else None,
            )

            # Track usage every round
            self.total_usage["input_tokens"] += response.usage.get("input_tokens", 0)
            self.total_usage["output_tokens"] += response.usage.get("output_tokens", 0)

            # If no tool calls, we're done
            if not response.tool_calls:
                break

            # Append assistant message with tool calls
            llm_messages.append(Message(
                role="assistant",
                content=response.content or "",
                tool_calls=response.tool_calls,
            ))

            # Execute each tool call and append results
            for tc in response.tool_calls:
                tool_name = tc.get("name", "")
                tool_id = tc.get("id", "")
                raw_args = tc.get("arguments", {})
                # arguments may be a JSON string from some providers
                if isinstance(raw_args, str):
                    try:
                        raw_args = json.loads(raw_args)
                    except json.JSONDecodeError:
                        raw_args = {}

                logger.info(f"Tool call: {tool_name}({raw_args})")
                result = await self.tools.execute(tool_name, raw_args)

                llm_messages.append(Message(
                    role="tool",
                    content=result,
                    tool_call_id=tool_id,
                    name=tool_name,
                ))

        # Add final assistant response to conversation history
        assert response is not None
        self.messages.append(Message(role="assistant", content=response.content))

        # Save transcript
        self._save_turn(user_input, response)

        # Background: tick extractor
        self.extractor.tick()

        # Background: try memory extraction (non-blocking)
        if self.extractor.should_extract():
            asyncio.create_task(self._run_extraction())

        # Background: check dream gates (non-blocking)
        if self.dream.should_dream():
            asyncio.create_task(self._run_dream())

        return response.content

    async def _run_extraction(self) -> None:
        """Run memory extraction in the background."""
        try:
            saved = await self.extractor.extract(
                recent_messages=self.messages[-10:],
                router=self.router,
            )
            if saved:
                logger.info(f"Background extraction saved {len(saved)} memories")
        except Exception as e:
            logger.error(f"Background extraction failed: {e}")

    async def _run_dream(self) -> None:
        """Run dream consolidation in the background."""
        try:
            result = await self.dream.run_dream(router=self.router)
            if result["status"] == "completed":
                logger.info(f"Dream completed: {result.get('summary', '')[:100]}")
        except Exception as e:
            logger.error(f"Dream failed: {e}")

    def _save_turn(self, user_input: str, response: LLMResponse) -> None:
        """Save conversation turn to transcript."""
        transcript_dir = Path(self.config.conversations.transcript_dir)
        transcript_file = transcript_dir / f"{self.session_id}.jsonl"

        turn = {
            "timestamp": datetime.now().isoformat(),
            "turn": self.turn_count,
            "user": user_input,
            "assistant": response.content,
            "model": response.model,
            "usage": response.usage,
        }

        with open(transcript_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(turn) + "\n")

    def get_status(self) -> dict:
        """Get agent status overview."""
        return {
            "session_id": self.session_id,
            "turns": self.turn_count,
            "usage": self.total_usage,
            "memory": self.memory_dir.get_stats(),
            "dream": self.dream.get_status(),
            "coordinator": self.coordinator.get_status(),
        }

    async def coordinate(self, task: str, on_progress=None) -> str:
        """Run a task through the multi-agent coordinator."""
        return await self.coordinator.coordinate(task, on_progress=on_progress)

    async def force_dream(self) -> dict:
        """Manually trigger a dream consolidation (bypasses time/session gates)."""
        return await self.dream.run_dream(router=self.router)

    async def force_extract(self) -> list:
        """Manually trigger memory extraction on recent messages."""
        return await self.extractor.extract(
            recent_messages=self.messages[-20:],
            router=self.router,
        )
