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

echo -e "${CYAN}[*] Starting Poetry installation...${RESET}"

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
    fi
fi

# Install Poetry
echo -e "${CYAN}[*] Downloading Poetry...${RESET}"
curl -sSL https://install.python-poetry.org | python3 -

POETRY_PATH="$HOME/.local/bin"

# Detect shell config
if [[ "$SHELL" == *"bash"* ]]; then
    SHELL_RC="$HOME/.bashrc"
elif [[ "$SHELL" == *"zsh"* ]]; then
    SHELL_RC="$HOME/.zshrc"
else
    SHELL_RC="$HOME/.profile"
fi

# Add PATH permanently
if ! grep -q "$POETRY_PATH" "$SHELL_RC" 2>/dev/null; then
    echo -e "${CYAN}[*] Adding Poetry to PATH -> $SHELL_RC${RESET}"
    echo "export PATH=\"$POETRY_PATH:\$PATH\"" >> "$SHELL_RC"
fi

# Temporary PATH
export PATH="$POETRY_PATH:$PATH"

echo -e "${CYAN}[*] Installation completed!${RESET}"

# Verify installation
if command -v poetry >/dev/null 2>&1; then
    echo -e "${CYAN}[+] Poetry version:${RESET}"
    poetry --version
else
    echo -e "${CYAN}[!] Poetry not found in PATH. Restart your terminal.${RESET}"
fi

if command -v poetry >/dev/null 2>&1; then
    echo -e "${CYAN}[+] Poetry version:${RESET}"
    poetry --version

    echo -e "${CYAN}[+] Download complete.${RESET}"
    echo -e "${CYAN}[*] Please start:${RESET}"
    echo -e "${CYAN}    poetry run sondra -h${RESET}"
else
    echo -e "${CYAN}[!] Poetry not found in PATH. Restart your terminal.${RESET}"
fi
