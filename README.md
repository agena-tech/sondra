<p align="center">
  <img src="docs/sondra.png" alt="Sondra Autonomous Agent Banner" width="100%">
</p>

# 🔰 SONDRA

### Unified Autonomous AI Agent for Research, Security, and Intelligent Automation

**An open-source platform for multi-domain autonomous agent workflows.**

<p align="center">
  <br />
  <a href="https://www.instagram.com/xx___xxbora_anezatraxx___xx_x/">
    <img src="https://img.shields.io/badge/Instagram-Anezatra-E4405F?logo=instagram&logoColor=white&style=for-the-badge" alt="Instagram">
  </a>
  <a href="https://www.youtube.com/@anezatra_official">
    <img src="https://img.shields.io/badge/YouTube-Anezatra_Official-FF0000?logo=youtube&logoColor=white&style=for-the-badge" alt="YouTube">
  </a>
  <a href="https://t.me/anezatra">
    <img src="https://img.shields.io/badge/Telegram-Anezatra-26A5E4?logo=telegram&logoColor=white&style=for-the-badge" alt="Telegram">
  </a>
  <br />
</p>

---

## What is Sondra?

Sondra is a unified autonomous AI agent platform designed for users who need one
powerful environment instead of fragmented tools for research, security testing,
automation, memory, terminal operations, browser control, and long-running tasks.

Its purpose is simple:

> One agent platform to think, act, remember, investigate, automate, and report.

Sondra is derived from the original STRIX project and extended by Agena Memory
Systems with a redesigned memory layer, adaptive terminal interface, emotional
context signals, task scheduling, voice output, and broader general-purpose
agent workflows.

---

## Purpose

Modern technical work is often split across many disconnected tools:

- browsers
- terminals
- scripts
- scanners
- AI chat tools
- notes
- memory systems
- automation pipelines

Sondra brings these pieces into a single agentic workspace.

It is built to operate as a practical AI operator: not only a chatbot, not only a
scanner, and not only a terminal wrapper, but a memory-aware autonomous agent
that can move between reasoning, execution, automation, and reporting.

---

## Capabilities

Sondra can:

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

## Memory System

Sondra includes a persistent memory system designed for long-term continuity.

The memory layer can store and reuse:

- Conversation history
- Semantic facts
- User profile facts
- Task state and progress
- Scheduled tasks
- Episodic events
- Emotional signals
- Recent interaction context
- Retrieval and recall results

The goal of the memory system is to prevent every session from starting as a
blank page.

Sondra can build adaptive context from multiple memory layers, allowing it to
respond with better continuity, preserve task awareness, and improve workflows
over time.

---

## Core Modes

### General Mode

```bash
poetry run sondra -m general
````

General assistant mode for everyday tasks, memory, chat, scheduling, browser
actions, terminal operations, and adaptive workflows.

### OSINT Mode

```bash
poetry run sondra -m osint -l standard --instruction "Investigate example.com"
```

OSINT mode for public research, target investigation, and information gathering.

### Pentest Mode

```bash
poetry run sondra -m pentest -l deep --target https://example.com
```

Pentest mode for deeper security testing and bug hunting workflows.

### ADB Mode

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

Check the CLI:

```bash
poetry run sondra -h
```

Start Sondra:

```bash
poetry run sondra -m general
```

---

## Model Configuration

Sondra supports multiple model providers through environment configuration.

### OpenAI

```bash
export SONDRA_LLM="openai/gpt-5"
export LLM_API_KEY="your-openai-api-key"
export LLM_API_BASE="https://api.openai.com/v1"
```

### Anthropic

```bash
export SONDRA_LLM="anthropic/claude-3-7-sonnet"
export LLM_API_KEY="your-anthropic-api-key"
export LLM_API_BASE="https://api.anthropic.com/v1"
```

### Ollama

```bash
export SONDRA_LLM="ollama/qwen2.5:7b"
export LLM_API_BASE="http://127.0.0.1:11434"
```

### LM Studio

```bash
export SONDRA_LLM="openai/local-model"
export LLM_API_BASE="http://127.0.0.1:1234/v1"
export LLM_API_KEY="lm-studio"
```

### Llama.cpp

```bash
export SONDRA_LLM="openai/qwen2.5:7b"
export LLM_API_BASE="http://127.0.0.1:8000/v1"
export LLM_API_KEY="local"
```

---

## Voice and Audio Setup

Sondra can speak assistant replies through local audio playback when voice mode
is enabled.

### WSL / Windows

If you are running Sondra inside WSL or a Windows-based environment, configure
`ffplay`:

```bash
echo 'export FFPLAY_COMMAND_DIR="/mnt/c/ffmpeg/bin/ffplay.exe"' >> ~/.profile
source ~/.profile
```

### Linux

On native Linux systems:

```bash
sudo apt install ffmpeg
```

Enable voice mode:

```bash
poetry run sondra -m general --voice-speech
```

---

## Auto Mode

Sondra can infer the mode, level, target, and instruction from natural language.

```bash
poetry run sondra -a "Open https://example.com and analyze the site"
```

Auto mode is useful when you want Sondra to decide whether a request should be
handled as a general task, OSINT workflow, ADB automation, or pentest workflow.

---

## Interactive TUI

Sondra includes an interactive terminal interface designed to make agent behavior
visible while it works.

The TUI includes:

* Main conversation terminal
* Agent map with root agent and subagents
* Subsystem menu
* Task panel
* Memory layer status
* Emotion indicators
* Request and response telemetry
* Stop and resume controls

The goal is not only to run agents, but to make their decisions, progress, and
tool usage easier to observe.

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

## Documentation

Full usage guide:

```text
docs/usage.mdx
docs/usage.pdf
```

---

## License and Origin

Sondra is based on the original STRIX project by `usestrix/strix`.

It has been significantly extended with a redesigned architecture, including a
reworked memory system, broader multi-mode capabilities, voice integration,
adaptive terminal interface, emotional signal layer, and enhanced autonomous
agent workflows.

Sondra is licensed under the Apache 2.0 License.

---

## Contact

**Agena Memory Systems**

Prepared by **Anezatra**

Email: `anezatra@gmail.com`
