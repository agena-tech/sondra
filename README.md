# 🔰 SONDRA  
### Unified Autonomous AI Agent for Research, Security, and Intelligent Automation

**An open-source platform for multi-domain autonomous agent workflows.**

Sondra is a unified AI agent system designed for users who need a single,
powerful environment instead of fragmented tools for research, security testing,
automation, memory, terminal operations, browser control, and long-running tasks.

Its first purpose is simple:

> One agent platform for everything you need: think, act, remember, investigate,
> automate, and report.

Sondra is derived from the STRIX project and extended by Agena Memory Systems
with a new memory layer, adaptive TUI, emotion-aware context, task scheduling,
voice output, and broader general-purpose workflows.

---

## What Sondra Can Do

- Run general autonomous assistant workflows.
- Perform OSINT research and investigation.
- Execute penetration testing and bug hunting workflows.
- Control Android devices through ADB task instructions.
- Use browser, terminal, Python, notes, file editing, reporting, and search tools.
- Spawn subagents for complex multi-step work.
- Remember user facts, preferences, context, emotions, tasks, and past events.
- Build adaptive context from semantic, episodic, profile, task, and recent memory.
- Speak assistant replies with local TTS when voice mode is enabled.
- Run in interactive TUI mode or non-interactive automation mode.

---

## Core Modes

```bash
poetry run sondra -m general
```

General assistant mode for everyday tasks, memory, chat, scheduling, browser
actions, and adaptive workflows.

```bash
poetry run sondra -m osint -l standard --instruction "Investigate example.com"
```

OSINT mode for public research, target investigation, and information gathering.

```bash
poetry run sondra -m pentest -l deep --target https://example.com
```

Pentest mode for deeper security testing and bug hunting workflows.

```bash
poetry run sondra -m adb -l standard --instruction-file ./adb_steps.txt
```

ADB mode for Android device automation through instruction files.

---

## Quick Start

Install dependencies:

```bash
bash install.sh
```

Configure your model:

Openai: 

```bash
export SONDRA_LLM="openai/gpt-5"
export LLM_API_KEY="your-openai-api-key"
export LLM_API_BASE="https://api.openai.com/v1"
```

Antrophic:

```bash
export SONDRA_LLM="anthropic/claude-3-7-sonnet"
export LLM_API_KEY="your-anthropic-api-key"
export LLM_API_BASE="https://api.anthropic.com/v1"
```

Ollama:

```bash
export SONDRA_LLM="ollama/qwen2.5:7b"
export LLM_API_BASE="http://127.0.0.1:11434"
```

Lm studio:

```bash
export SONDRA_LLM="openai/local-model"
export LLM_API_BASE="http://127.0.0.1:1234/v1"
export LLM_API_KEY="lm-studio"
```

Llama-cpp:

```bash
export SONDRA_LLM="openai/qwen2.5:7b"
export LLM_API_BASE="http://127.0.0.1:8000/v1"
export LLM_API_KEY="local"
```

For WSL or Windows audio playback, configure `ffplay`:

```bash
echo 'export FFPLAY_COMMAND_DIR="/mnt/c/ffmpeg/bin/ffplay.exe"' >> ~/.profile
source ~/.profile
```

On native Linux:

```bash
sudo apt install ffmpeg
```

Check the CLI:

```bash
poetry run sondra -h
```

Start Sondra:

```bash
poetry run sondra -m general
```

---

## Auto Mode

Sondra can infer the mode, level, target, and instruction from natural language.

```bash
poetry run sondra -a "Open https://example.com and analyze the site"
```

Auto mode is useful when you want Sondra to decide whether a request should be
general, OSINT, ADB, or pentest oriented.

---

## Memory System

Sondra includes a persistent memory system designed for continuity.

It can store and reuse:

- Conversation history
- Semantic facts
- User profile facts
- Scheduled tasks
- Episodic events
- Emotional signals
- Recent context
- Retrieval and recall results

The memory layer helps Sondra respond with better context over time instead of
treating every session as a blank page.

---

## Interactive TUI

The terminal UI gives live visibility into what the agent is doing.

It includes:

- Main conversation terminal
- Agent map with root agent and subagents
- Subsystem menu
- Task panel
- Memory layer status
- Emotion indicators
- Request and response telemetry
- Stop/resume controls

The goal is not only to run agents, but to make their behavior visible while
they work.

---

## Voice And Audio

Enable spoken assistant replies:

```bash
poetry run sondra -m general --voice-speech
```

---

## Useful Options

```bash
--instruction "task text"
--instruction-file ./task.txt
--subagents 2
--retry 5
--voice-speech
-n, --non-interactive
-a, --auto "natural language request"
```

---

## Why Sondra Exists

Modern work is fragmented across browsers, shells, scripts, notes, scanners,
LLMs, memory systems, and automation tools.

Sondra brings those pieces into one agentic environment.

It is built to be practical: not just a chatbot, not just a scanner, not just a
terminal wrapper, but a memory-aware AI operator that can move between research,
execution, automation, and reporting.

---

## Documentation

Full usage guide:

```text
docs/usage.mdx
docs/usage.pdf
```

---

## License And Origin

Sondra is derived from the STRIX project and is licensed under the Apache 2.0
License.

Additional capabilities, including the intelligent memory system, user interface
design, emotional signal layer, task behavior, speech integration, and extended
general workflows, were developed by Agena Memory Systems.

---

## Contact

**Agena Memory Systems**

Prepared by **Anezatra**

Email: `anezatra@gmail.com`
