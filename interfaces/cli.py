"""Rich terminal CLI for Sutra."""

import asyncio
import logging
import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory

logger = logging.getLogger(__name__)

theme = Theme({
    "user": "bold cyan",
    "assistant": "bold green",
    "system": "dim yellow",
    "error": "bold red",
})

console = Console(theme=theme)


BANNER = r"""
███████╗██╗   ██╗████████╗██████╗  █████╗
██╔════╝██║   ██║╚══██╔══╝██╔══██╗██╔══██╗
███████╗██║   ██║   ██║   ██████╔╝███████║
╚════██║██║   ██║   ██║   ██╔══██╗██╔══██║
███████║╚██████╔╝   ██║   ██║  ██║██║  ██║
╚══════╝ ╚═════╝    ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝
  Threads of Knowledge — by Sumanth Reddy
"""

HELP_TEXT = """
Commands:
  /status      — Show agent status (memory, dream, vectors, usage)
  /memory      — Show memory index
  /memories    — List all memory files
  /search      — Semantic search across memories
  /reindex     — Re-embed all memories into vector store
  /coordinate  — Break a complex task into parallel sub-tasks
  /goal        — Create or list goals (/goal new <title> or /goal list)
  /summary     — Generate daily summary
  /patterns    — Show behavioral patterns
  /predict     — Get predictions about what you might need
  /dream       — Force a dream consolidation
  /extract     — Force memory extraction
  /voice       — Toggle voice mode (speak responses aloud)
  /listen      — Record from mic and send as message
  /clear       — Clear conversation (keeps memory)
  /help        — Show this help
  /quit        — Exit Sutra
"""


class CLI:
    """Rich terminal interface for Sutra."""

    def __init__(self, agent):
        self.agent = agent
        self.voice_mode = False
        self.prompt_session = PromptSession(
            history=FileHistory(".sutra_history"),
        )

    def show_banner(self) -> None:
        console.print(BANNER, style="bold blue")
        console.print("  Type [bold cyan]/help[/] for commands, or just start chatting.\n")

    async def handle_command(self, command: str) -> bool:
        """Handle slash commands. Returns True if handled."""
        cmd = command.strip().lower()

        if cmd == "/quit" or cmd == "/exit":
            console.print("\n[system]Goodbye! Your memories persist.[/]")
            return True

        if cmd == "/help":
            console.print(HELP_TEXT, style="system")
            return False

        if cmd == "/status":
            status = self.agent.get_status()
            table = Table(title="Sutra Status")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")
            table.add_row("Session", status["session_id"])
            table.add_row("Turns", str(status["turns"]))
            table.add_row("Tokens (in/out)", f"{status['usage']['input_tokens']}/{status['usage']['output_tokens']}")
            table.add_row("Memories", str(status["memory"]["total_memories"]))
            table.add_row("Index lines", str(status["memory"]["index_lines"]))
            dream = status["dream"]
            table.add_row("Last dream", dream["last_consolidation"])
            table.add_row("Dream ready", str(dream["ready"]))
            table.add_row("Sessions since dream", f"{dream['sessions_since']}/{dream['min_sessions']}")
            if self.agent.vector_store:
                vs = self.agent.vector_store.get_stats()
                table.add_row("Vector embeddings", str(vs.get("total_embeddings", 0)))
                table.add_row("Embedding provider", vs.get("embedding_provider", "unknown"))
            coord = status.get("coordinator", {})
            table.add_row("Coordinator active", str(coord.get("active", False)))
            console.print(table)
            return False

        if cmd == "/memory":
            index = self.agent.memory_dir.read_index()
            if index:
                console.print(Panel(Markdown(index), title="MEMORY.md"))
            else:
                console.print("[system]Memory index is empty.[/]")
            return False

        if cmd == "/memories":
            memories = self.agent.memory_dir.list_memories()
            if not memories:
                console.print("[system]No memories saved yet.[/]")
                return False
            table = Table(title=f"Memories ({len(memories)} files)")
            table.add_column("File", style="cyan")
            table.add_column("Type", style="yellow")
            table.add_column("Name", style="green")
            table.add_column("Description")
            for m in memories:
                table.add_row(m.path, m.type.value, m.name, m.description[:50])
            console.print(table)
            return False

        if cmd == "/dream":
            console.print("[system]Running dream consolidation...[/]")
            result = await self.agent.force_dream()
            console.print(Panel(
                result.get("summary", result.get("error", "No changes")),
                title=f"Dream [{result['status']}]",
            ))
            return False

        if cmd == "/extract":
            console.print("[system]Running memory extraction...[/]")
            saved = await self.agent.force_extract()
            if saved:
                for m in saved:
                    console.print(f"  [green]✓[/] {m.name} ({m.type.value})")
            else:
                console.print("[system]Nothing to extract.[/]")
            return False

        if cmd == "/clear":
            self.agent.messages.clear()
            self.agent.turn_count = 0
            console.print("[system]Conversation cleared. Memory persists.[/]")
            return False

        if cmd == "/reindex":
            if not self.agent.vector_store:
                console.print("[system]Vector store not configured.[/]")
                return False
            console.print("[system]Re-embedding all memories...[/]")
            memories = self.agent.memory_dir.list_memories()
            count = self.agent.vector_store.reindex_all(memories)
            console.print(f"[green]Reindexed {count}/{len(memories)} memories.[/]")
            return False

        if cmd.startswith("/search"):
            query = cmd[7:].strip()
            if not query:
                console.print("[system]Usage: /search <query>[/]")
                return False
            results = self.agent.memory_dir.semantic_search(query, n_results=5)
            if not results:
                console.print(f"[system]No memories matching '{query}'.[/]")
                return False
            table = Table(title=f"Search: {query}")
            table.add_column("Score", style="yellow")
            table.add_column("Name", style="green")
            table.add_column("Type", style="cyan")
            table.add_column("Description")
            for r in results:
                table.add_row(
                    str(r["score"]),
                    r["name"],
                    r["type"],
                    r["description"][:60],
                )
            console.print(table)
            return False

        if cmd.startswith("/coordinate"):
            task = cmd[11:].strip()
            if not task:
                console.print("[system]Usage: /coordinate <complex task description>[/]")
                return False
            console.print(f"[system]Coordinating: {task[:80]}...[/]")

            def on_progress(plan):
                console.print(f"[dim]  Progress: {plan.progress}[/]")

            with console.status("[dim]Sub-agents working...[/]"):
                result = await self.agent.coordinate(task, on_progress=on_progress)
            console.print()
            console.print(Panel(Markdown(result), title="Coordination Result", border_style="green"))
            if self.voice_mode and self.agent.voice_out:
                console.print("[dim]Speaking...[/]")
                await self.agent.voice_out.speak_and_play(result)
            console.print()
            return False

        if cmd == "/voice":
            self.voice_mode = not self.voice_mode
            state = "ON" if self.voice_mode else "OFF"
            console.print(f"[system]Voice mode: {state}[/]")
            if self.voice_mode and not self.agent.voice_out:
                console.print("[system]Warning: Voice output not configured.[/]")
            return False

        if cmd == "/listen":
            if not self.agent.voice_in:
                console.print("[system]Voice input not configured.[/]")
                return False
            console.print("[system]Listening... (5 seconds)[/]")
            text = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.agent.voice_in.record_and_transcribe(duration=5.0),
            )
            if text.startswith("Error:"):
                console.print(f"[error]{text}[/]")
                return False
            console.print(f"[user]You (voice):[/] {text}")
            # Process as a normal message
            with console.status("[dim]Thinking...[/]"):
                response = await self.agent.chat(text)
            console.print()
            console.print(Panel(Markdown(response), title="Sutra", border_style="green"))
            if self.voice_mode and self.agent.voice_out:
                console.print("[dim]Speaking...[/]")
                await self.agent.voice_out.speak_and_play(response)
            console.print()
            return False

        if cmd.startswith("/goal"):
            args = cmd[5:].strip()
            if args.startswith("new "):
                title = args[4:].strip()
                if not title:
                    console.print("[system]Usage: /goal new <title>[/]")
                    return False
                goal = self.agent.goals.create_goal(title)
                console.print(f"[green]Created goal: {goal.title} (0%)[/]")
            elif args.startswith("progress "):
                parts = args[9:].strip().split(" ", 1)
                if len(parts) < 1:
                    console.print("[system]Usage: /goal progress <id> <percent>[/]")
                    return False
                goal_id = parts[0]
                progress = int(parts[1]) if len(parts) > 1 else 0
                result = self.agent.goals.update_progress(goal_id, progress)
                if result:
                    console.print(f"[green]Updated: {result.title} → {result.progress}%[/]")
                else:
                    console.print(f"[error]Goal not found: {goal_id}[/]")
            else:
                goals = self.agent.goals.list_goals()
                if not goals:
                    console.print("[system]No goals set. Use /goal new <title>[/]")
                    return False
                table = Table(title="Goals")
                table.add_column("ID", style="cyan")
                table.add_column("Title", style="green")
                table.add_column("Progress", style="yellow")
                table.add_column("Status")
                table.add_column("Deadline")
                for g in goals:
                    table.add_row(g.id, g.title, f"{g.progress}%", g.status, g.deadline or "-")
                console.print(table)
            return False

        if cmd == "/summary":
            console.print("[system]Generating daily summary...[/]")
            with console.status("[dim]Summarizing...[/]"):
                summary = await self.agent.proactive.generate_daily_summary(self.agent.router)
            console.print(Panel(Markdown(summary), title="Daily Summary", border_style="blue"))
            return False

        if cmd == "/patterns":
            summary = self.agent.pattern_detector.get_summary(days=7)
            console.print(Panel(summary, title="Behavioral Patterns (7 days)", border_style="magenta"))
            return False

        if cmd == "/predict":
            console.print("[system]Generating predictions...[/]")
            with console.status("[dim]Thinking...[/]"):
                preds = await self.agent.predictions.predict(
                    self.agent.router,
                    last_messages=self.agent.messages[-10:],
                )
            if preds:
                for p in preds:
                    console.print(f"  [magenta]🔮[/] {p}")
            else:
                console.print("[system]No strong predictions right now.[/]")
            return False

        console.print(f"[error]Unknown command: {command}[/]")
        return False

    async def run(self) -> None:
        """Main REPL loop."""
        self.show_banner()

        while True:
            try:
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.prompt_session.prompt("You → "),
                )

                if not user_input.strip():
                    continue

                # Handle commands
                if user_input.startswith("/"):
                    should_exit = await self.handle_command(user_input)
                    if should_exit:
                        break
                    continue

                # Chat
                console.print(f"\n[user]You:[/] {user_input}")
                with console.status("[dim]Thinking...[/]"):
                    response = await self.agent.chat(user_input)

                console.print()
                console.print(Panel(Markdown(response), title="Sutra", border_style="green"))
                if self.voice_mode and self.agent.voice_out:
                    console.print("[dim]Speaking...[/]")
                    await self.agent.voice_out.speak_and_play(response)
                console.print()

            except KeyboardInterrupt:
                console.print("\n[system]Use /quit to exit.[/]")
            except EOFError:
                break
            except Exception as e:
                console.print(f"[error]Error: {e}[/]")
                logger.exception("CLI error")
