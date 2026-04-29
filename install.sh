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

# ------------------------------------------------------------
# Helper: install package
# ------------------------------------------------------------
install_package() {
    local package_name="$1"

    if command -v apt >/dev/null 2>&1; then
        sudo apt update
        sudo apt install -y "$package_name"
    elif command -v dnf >/dev/null 2>&1; then
        sudo dnf install -y "$package_name"
    elif command -v pacman >/dev/null 2>&1; then
        sudo pacman -Sy --noconfirm "$package_name"
    else
        echo -e "${CYAN}[!] No supported package manager found. Install $package_name manually.${RESET}"
        exit 1
    fi
}

# ------------------------------------------------------------
# Check Python
# ------------------------------------------------------------
if ! command -v python3 >/dev/null 2>&1; then
    echo -e "${CYAN}[!] python3 not found, installing...${RESET}"

    if command -v apt >/dev/null 2>&1; then
        sudo apt update
        sudo apt install -y python3 python3-pip curl
    elif command -v dnf >/dev/null 2>&1; then
        sudo dnf install -y python3 python3-pip curl
    elif command -v pacman >/dev/null 2>&1; then
        sudo pacman -Sy --noconfirm python python-pip curl
    else
        echo -e "${CYAN}[!] No supported package manager found. Install python3 manually.${RESET}"
        exit 1
    fi
fi

# ------------------------------------------------------------
# Check curl
# ------------------------------------------------------------
if ! command -v curl >/dev/null 2>&1; then
    echo -e "${CYAN}[!] curl not found, installing...${RESET}"
    install_package curl
fi

# ------------------------------------------------------------
# Check zstd
# ------------------------------------------------------------
echo -e "${CYAN}[*] Checking zstd...${RESET}"

if ! command -v zstd >/dev/null 2>&1; then
    echo -e "${CYAN}[!] zstd not found, installing...${RESET}"
    install_package zstd
else
    echo -e "${CYAN}[+] zstd already installed.${RESET}"
fi

# ------------------------------------------------------------
# Poetry PATH setup
# ------------------------------------------------------------
POETRY_PATH="$HOME/.local/bin"
export PATH="$POETRY_PATH:/usr/local/bin:$PATH"

if [[ "$SHELL" == *"bash"* ]]; then
    SHELL_RC="$HOME/.bashrc"
elif [[ "$SHELL" == *"zsh"* ]]; then
    SHELL_RC="$HOME/.zshrc"
else
    SHELL_RC="$HOME/.profile"
fi

# Add Poetry PATH permanently
if ! grep -q 'export PATH="$HOME/.local/bin:$PATH"' "$SHELL_RC" 2>/dev/null; then
    echo -e "${CYAN}[*] Adding Poetry PATH permanently -> $SHELL_RC${RESET}"
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$SHELL_RC"
fi

# ------------------------------------------------------------
# Install Poetry
# ------------------------------------------------------------
echo -e "${CYAN}[*] Checking Poetry...${RESET}"

if ! command -v poetry >/dev/null 2>&1; then
    echo -e "${CYAN}[!] Poetry not found, installing...${RESET}"

    if command -v apt >/dev/null 2>&1; then
        echo -e "${CYAN}[*] Trying to install Poetry with apt...${RESET}"
        sudo apt update
        sudo apt install -y python3-poetry || true
    fi

    export PATH="$POETRY_PATH:/usr/local/bin:$PATH"

    if ! command -v poetry >/dev/null 2>&1; then
        echo -e "${CYAN}[!] Poetry still not found after apt install.${RESET}"
        echo -e "${CYAN}[*] Trying official Poetry installer...${RESET}"
        curl -sSL https://install.python-poetry.org | python3 -
    fi

    export PATH="$POETRY_PATH:/usr/local/bin:$PATH"

    # If Poetry exists in user local bin, link it globally
    if [ -x "$POETRY_PATH/poetry" ]; then
        echo -e "${CYAN}[*] Linking Poetry to /usr/local/bin/poetry...${RESET}"
        sudo ln -sf "$POETRY_PATH/poetry" /usr/local/bin/poetry
    fi

    # If apt installed poetry somewhere else, keep it as command
    if ! command -v poetry >/dev/null 2>&1 && [ -x /usr/bin/poetry ]; then
        echo -e "${CYAN}[*] Linking apt Poetry to /usr/local/bin/poetry...${RESET}"
        sudo ln -sf /usr/bin/poetry /usr/local/bin/poetry
    fi

    export PATH="/usr/local/bin:$POETRY_PATH:$PATH"

    # Final retry if still missing
    if ! command -v poetry >/dev/null 2>&1; then
        echo -e "${CYAN}[!] Poetry command still not found. Retrying official installer...${RESET}"
        curl -sSL https://install.python-poetry.org | python3 -
        export PATH="/usr/local/bin:$POETRY_PATH:$PATH"

        if [ -x "$POETRY_PATH/poetry" ]; then
            sudo ln -sf "$POETRY_PATH/poetry" /usr/local/bin/poetry
        fi
    fi

    if ! command -v poetry >/dev/null 2>&1; then
        echo -e "${CYAN}[!] Poetry installation failed. Command not found.${RESET}"
        echo -e "${CYAN}[!] Try manually: sudo apt install -y python3-poetry${RESET}"
        exit 1
    fi
else
    echo -e "${CYAN}[+] Poetry already installed.${RESET}"

    # Even if already installed, make sure it is globally available
    if [ -x "$POETRY_PATH/poetry" ]; then
        sudo ln -sf "$POETRY_PATH/poetry" /usr/local/bin/poetry
    elif [ -x /usr/bin/poetry ]; then
        sudo ln -sf /usr/bin/poetry /usr/local/bin/poetry
    fi
fi

echo -e "${CYAN}[+] Poetry installed:${RESET}"
poetry --version

# ------------------------------------------------------------
# Install Ollama
# ------------------------------------------------------------
echo -e "${CYAN}[*] Checking Ollama...${RESET}"

export PATH="/usr/local/bin:/usr/bin:$PATH"

if ! command -v ollama >/dev/null 2>&1; then
    echo -e "${CYAN}[!] Ollama not found, installing...${RESET}"

    if command -v apt >/dev/null 2>&1; then
        echo -e "${CYAN}[*] Trying to install Ollama with apt...${RESET}"
        sudo apt update
        sudo apt install -y ollama || true
    fi

    export PATH="/usr/local/bin:/usr/bin:$PATH"

    if ! command -v ollama >/dev/null 2>&1; then
        echo -e "${CYAN}[!] Ollama still not found after apt install.${RESET}"
        echo -e "${CYAN}[*] Trying official Ollama installer...${RESET}"
        curl -fsSL https://ollama.com/install.sh | sh
    fi

    export PATH="/usr/local/bin:/usr/bin:$PATH"

    # Link common Ollama locations globally
    if [ -x /usr/local/bin/ollama ]; then
        sudo ln -sf /usr/local/bin/ollama /usr/bin/ollama 2>/dev/null || true
    elif [ -x /usr/bin/ollama ]; then
        sudo ln -sf /usr/bin/ollama /usr/local/bin/ollama 2>/dev/null || true
    elif [ -x "$HOME/.ollama/bin/ollama" ]; then
        sudo ln -sf "$HOME/.ollama/bin/ollama" /usr/local/bin/ollama
    fi

    export PATH="/usr/local/bin:/usr/bin:$PATH"

    # Final retry if still missing
    if ! command -v ollama >/dev/null 2>&1; then
        echo -e "${CYAN}[!] Ollama command still not found. Retrying official installer...${RESET}"
        curl -fsSL https://ollama.com/install.sh | sh

        if [ -x /usr/local/bin/ollama ]; then
            sudo ln -sf /usr/local/bin/ollama /usr/bin/ollama 2>/dev/null || true
        elif [ -x "$HOME/.ollama/bin/ollama" ]; then
            sudo ln -sf "$HOME/.ollama/bin/ollama" /usr/local/bin/ollama
        fi
    fi

    if ! command -v ollama >/dev/null 2>&1; then
        echo -e "${CYAN}[!] Ollama installation failed. Command not found.${RESET}"
        exit 1
    fi
else
    echo -e "${CYAN}[+] Ollama already installed.${RESET}"

    # Even if already installed, make sure it is globally available
    if [ -x /usr/local/bin/ollama ]; then
        sudo ln -sf /usr/local/bin/ollama /usr/bin/ollama 2>/dev/null || true
    elif [ -x /usr/bin/ollama ]; then
        sudo ln -sf /usr/bin/ollama /usr/local/bin/ollama 2>/dev/null || true
    fi
fi

echo -e "${CYAN}[+] Ollama installed:${RESET}"
ollama --version || true

# ------------------------------------------------------------
# Start Ollama server
# ------------------------------------------------------------
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

# Final Ollama server check
if ! ollama list >/dev/null 2>&1; then
    echo -e "${CYAN}[!] Ollama server is not responding.${RESET}"
    echo -e "${CYAN}[!] Check log: /tmp/ollama.log${RESET}"
    exit 1
fi

# ------------------------------------------------------------
# Pull embedding model
# ------------------------------------------------------------
echo -e "${CYAN}[*] Pulling nomic-embed-text model...${RESET}"
ollama pull nomic-embed-text

# ------------------------------------------------------------
# Final check
# ------------------------------------------------------------
echo -e "${CYAN}[*] Final command check...${RESET}"

if ! command -v poetry >/dev/null 2>&1; then
    echo -e "${CYAN}[!] Final check failed: poetry command not found.${RESET}"
    exit 1
fi

if ! command -v ollama >/dev/null 2>&1; then
    echo -e "${CYAN}[!] Final check failed: ollama command not found.${RESET}"
    exit 1
fi

echo -e "${CYAN}[+] Download complete.${RESET}"
echo -e "${CYAN}[+] Poetry:${RESET} $(poetry --version)"
echo -e "${CYAN}[+] Ollama:${RESET} $(ollama --version || true)"
echo -e "${CYAN}[*] Please start:${RESET}"
echo -e "${CYAN}    poetry run sondra -h${RESET}"
