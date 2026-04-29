#!/usr/bin/env bash

set -e

# Neon cyan color
CYAN="\033[1;36m"
RESET="\033[0m"

clear
echo -e "${CYAN}"
cat << "EOF"
_______  _____  __   _ ______   ______ _______
|______ |     | | \  | |     \ |_____/ |_____|
 ______||_____| |  \_| |_____/ |    \_ |     |

            ᴀɢᴇɴᴀ ᴍᴇᴍᴏʀʏ sʏsᴛᴇᴍs
            · · ──── ·⟡· ──── · ·
EOF
echo -e "${RESET}"

echo -e "${CYAN}[*] Starting installation...${RESET}"

# Check Python
if ! command -v python3 >/dev/null 2>&1; then
    echo -e "${CYAN}[!] python3 not found, installing...${RESET}"

    if command -v apt >/dev/null 2>&1; then
        sudo apt update && sudo apt install -y python3 python3-pip curl
    elif command -v dnf >/dev/null 2>&1; then
        sudo dnf install -y python3 python3-pip curl
    elif command -v pacman >/dev/null 2>&1; then
        sudo pacman -Sy --noconfirm python python-pip curl
    else
        echo -e "${CYAN}[!] No supported package manager found. Install python3 manually.${RESET}"
        exit 1
    fi
fi

# Check curl
if ! command -v curl >/dev/null 2>&1; then
    echo -e "${CYAN}[!] curl not found, installing...${RESET}"

    if command -v apt >/dev/null 2>&1; then
        sudo apt install -y curl
    elif command -v dnf >/dev/null 2>&1; then
        sudo dnf install -y curl
    elif command -v pacman >/dev/null 2>&1; then
        sudo pacman -Sy --noconfirm curl
    else
        echo -e "${CYAN}[!] No supported package manager found. Install curl manually.${RESET}"
        exit 1
    fi
fi

# Check zstd
echo -e "${CYAN}[*] Checking zstd...${RESET}"

if ! command -v zstd >/dev/null 2>&1; then
    echo -e "${CYAN}[!] zstd not found, installing...${RESET}"

    if command -v apt >/dev/null 2>&1; then
        sudo apt update && sudo apt install -y zstd
    elif command -v dnf >/dev/null 2>&1; then
        sudo dnf install -y zstd
    elif command -v pacman >/dev/null 2>&1; then
        sudo pacman -Sy --noconfirm zstd
    else
        echo -e "${CYAN}[!] No supported package manager found. Install zstd manually.${RESET}"
        exit 1
    fi
else
    echo -e "${CYAN}[+] zstd already installed.${RESET}"
fi

# Install Poetry
echo -e "${CYAN}[*] Checking Poetry...${RESET}"

if ! command -v poetry >/dev/null 2>&1; then
    echo -e "${CYAN}[!] Poetry not found, installing...${RESET}"

    POETRY_INSTALLED=false

    if command -v apt >/dev/null 2>&1; then
        echo -e "${CYAN}[*] Trying to install Poetry with apt...${RESET}"

        if sudo apt update && sudo apt install -y python3-poetry; then
            POETRY_INSTALLED=true
            echo -e "${CYAN}[+] Poetry installed with apt.${RESET}"
        else
            echo -e "${CYAN}[!] apt install python3-poetry failed.${RESET}"
        fi
    fi

    if [ "$POETRY_INSTALLED" = false ]; then
        echo -e "${CYAN}[*] Installing Poetry with official installer...${RESET}"

        if curl -sSL https://install.python-poetry.org | python3 -; then
            POETRY_INSTALLED=true
            echo -e "${CYAN}[+] Poetry installed with official installer.${RESET}"
        else
            echo -e "${CYAN}[!] Poetry installation failed.${RESET}"
            exit 1
        fi
    fi
else
    echo -e "${CYAN}[+] Poetry already installed.${RESET}"
fi

POETRY_PATH="$HOME/.local/bin"

# Detect shell config
if [[ "$SHELL" == *"bash"* ]]; then
    SHELL_RC="$HOME/.bashrc"
elif [[ "$SHELL" == *"zsh"* ]]; then
    SHELL_RC="$HOME/.zshrc"
else
    SHELL_RC="$HOME/.profile"
fi

# Add Poetry PATH permanently
if ! grep -q "$POETRY_PATH" "$SHELL_RC" 2>/dev/null; then
    echo -e "${CYAN}[*] Adding Poetry to PATH -> $SHELL_RC${RESET}"
    echo "export PATH=\"$POETRY_PATH:\$PATH\"" >> "$SHELL_RC"
fi

# Temporary PATH
export PATH="$POETRY_PATH:$PATH"

# Verify Poetry
if command -v poetry >/dev/null 2>&1; then
    echo -e "${CYAN}[+] Poetry installed:${RESET}"
    poetry --version
else
    echo -e "${CYAN}[!] Poetry not found in PATH. Restart your terminal.${RESET}"
fi

# Install Ollama
echo -e "${CYAN}[*] Checking Ollama...${RESET}"

if ! command -v ollama >/dev/null 2>&1; then
    echo -e "${CYAN}[!] Ollama not found, installing...${RESET}"

    OLLAMA_INSTALLED=false

    if command -v apt >/dev/null 2>&1; then
        echo -e "${CYAN}[*] Trying to install Ollama with apt...${RESET}"

        if sudo apt update && sudo apt install -y ollama; then
            OLLAMA_INSTALLED=true
            echo -e "${CYAN}[+] Ollama installed with apt.${RESET}"
        else
            echo -e "${CYAN}[!] apt install ollama failed.${RESET}"
        fi
    fi

    if [ "$OLLAMA_INSTALLED" = false ]; then
        echo -e "${CYAN}[*] Installing Ollama with official installer...${RESET}"

        if curl -fsSL https://ollama.com/install.sh | sh; then
            OLLAMA_INSTALLED=true
            echo -e "${CYAN}[+] Ollama installed with official installer.${RESET}"
        else
            echo -e "${CYAN}[!] Ollama installation failed.${RESET}"
            exit 1
        fi
    fi
else
    echo -e "${CYAN}[+] Ollama already installed.${RESET}"
fi

# Start Ollama server if needed
echo -e "${CYAN}[*] Starting Ollama service...${RESET}"

if command -v systemctl >/dev/null 2>&1; then
    sudo systemctl enable ollama >/dev/null 2>&1 || true
    sudo systemctl start ollama >/dev/null 2>&1 || true
fi

# Fallback for WSL / non-systemd environments
if ! pgrep -x "ollama" >/dev/null 2>&1; then
    echo -e "${CYAN}[*] Running ollama serve in background...${RESET}"
    nohup ollama serve >/tmp/ollama.log 2>&1 &
    sleep 3
fi

# Pull embedding model
echo -e "${CYAN}[*] Pulling nomic-embed-text model...${RESET}"
ollama pull nomic-embed-text

echo -e "${CYAN}[+] Download complete.${RESET}"
echo -e "${CYAN}[*] Please start:${RESET}"
echo -e "${CYAN}    poetry run sondra -h${RESET}"
