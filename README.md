<div align="center">

# सूत्र — Sutra

**Threads of Knowledge**

A personal AI agent with persistent memory, dream consolidation, and multi-modal awareness.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

</div>

---

Most AI assistants forget you the moment the conversation ends. Sutra doesn't.

It stores memories as files, consolidates them through a dream-like process while idle, and retrieves them semantically when relevant. Over time, it learns your patterns, tracks your goals, and starts acting before you ask.

## Features

- **Persistent Memory** — Memories stored as markdown files with YAML frontmatter. Survives restarts, searchable by meaning.
- **Dream Consolidation** — A 4-phase background process (Orient → Gather → Consolidate → Prune) that compresses scattered memories into durable knowledge, inspired by how human sleep consolidates learning.
- **Semantic Search** — ChromaDB vector embeddings with hybrid scoring that balances meaning, recency, and access frequency.
- **Multi-Modal** — Voice input via Whisper, text-to-speech via edge-tts, image understanding via Claude/GPT-4o vision.
- **Multi-Agent Coordination** — Decomposes complex tasks into subtasks, executes them in parallel with up to 4 workers, and synthesizes results.
- **Proactive Behavior** — Scheduled tasks, file watchers, idle detection, daily summaries, and goal tracking across sessions.
- **Pattern Recognition** — Detects behavioral patterns over weeks of usage. Predicts what you'll need before you ask.
- **Smart Routing** — Routes simple tasks to local models (Ollama, free) and complex ones to cloud (Anthropic/OpenAI) automatically.

## Getting Started

```bash
git clone https://github.com/sumanthreddyy/Sutra.git
cd Sutra
pip install -r requirements.txt
```

Copy the example config and add at least one API key:

```bash
cp config.example.yaml data/config.yaml
```

```yaml
providers:
  anthropic:
    api_key: "sk-..."        # or set ANTHROPIC_API_KEY env var
  openai:
    api_key: "sk-..."        # or set OPENAI_API_KEY env var
  ollama:
    base_url: "http://localhost:11434"  # free, local, no key needed
```

Run:

```bash
python main.py
```

You only need **one** provider. Ollama works fully offline and free.

## Usage

Sutra runs as a terminal CLI. Talk to it naturally, or use commands:

```
/memory        Browse stored memories
/search <q>    Search memories by meaning
/dream         Trigger memory consolidation
/voice         Toggle voice output
/listen        Speak instead of typing
/coordinate    Multi-agent mode for complex tasks
/goal new <t>  Set a goal to track across sessions
/summary       Generate a daily summary
/patterns      View detected behavioral patterns
/predict       See what Sutra thinks you'll need next
```

## How It Works

```
You speak → Agent processes → Tools execute → Memory stores → Dreams consolidate
                ↓                                    ↑
          Routes to best LLM              Semantic search retrieves
          (local or cloud)                relevant past context
```

**Memory** lives as markdown files organized by topic, indexed in a central `MEMORY.md`. A vector store (ChromaDB) enables search by meaning rather than just keywords.

**Dreams** run when the agent is idle. They scan recent conversations, extract facts worth keeping, merge duplicates, and prune stale information — similar to how sleep consolidates memory in the brain.

**Coordination** breaks complex requests into a dependency graph of subtasks, runs independent ones in parallel, and feeds results forward to dependent tasks.

## Architecture

```
sutra/
├── core/           Agent loop, LLM router, prompt builder
├── memory/         Dream engine, memory index, vector store, extractor
├── providers/      Anthropic, OpenAI, Ollama (unified interface)
├── tools/          Shell, file ops, web search, memory tools
├── senses/         Voice I/O, vision, screenshots
├── coordination/   Task decomposition, parallel workers, shared context
├── autonomy/       Scheduler, file watchers, proactive engine, goals
├── intuition/      Pattern detection, predictions
├── interfaces/     Rich terminal CLI
└── data/           Memories, vectors, transcripts (git-ignored)
```

Each package maps to a version, and each version maps to a [chakra](ROADMAP.md) — an ascending framework where each layer depends on the ones below it.

## Roadmap

See [ROADMAP.md](ROADMAP.md) for the full plan.

```
✅ v0.1  Root         — Memory, agent loop, tools
✅ v0.2  Sacral       — Semantic search, vector embeddings
✅ v0.3  Solar Plexus — Voice, vision, multi-modal senses
✅ v0.4  Heart        — Multi-agent coordination
✅ v0.5  Throat       — Autonomy, scheduling, goals
✅ v0.6  Third Eye    — Pattern detection, predictions
⚪ v1.0  Crown        — ...
```

## Contributing

This is a personal project, but ideas and feedback are welcome. Open an issue or start a discussion.

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.

---

<div align="center">
<sub>Built by <a href="https://github.com/sumanthreddyy">Sumanth Reddy</a></sub>
</div>
