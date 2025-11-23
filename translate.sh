#!/bin/sh
#
# Movie Translator - Sequential File-by-File Workflow
# POSIX-compliant shell script (works with sh, bash, zsh, etc.)
# Process each MKV file completely before moving to the next
#
# Auto-installs dependencies on macOS:
#   - Homebrew (if needed)
#   - uv (Python package manager)
#   - mkvtoolnix (for mkvmerge/mkvextract)
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Function to print colored messages
log_info() {
    printf "${BLUE}[INFO]${NC} %s\n" "$1"
}

log_success() {
    printf "${GREEN}[SUCCESS]${NC} %s\n" "$1"
}

log_warning() {
    printf "${YELLOW}[WARNING]${NC} %s\n" "$1"
}

log_error() {
    printf "${RED}[ERROR]${NC} %s\n" "$1"
}

log_progress() {
    printf "${CYAN}%s${NC}\n" "$1"
}

# Dependency checking and installation functions
check_python() {
    if ! command -v python3 >/dev/null 2>&1; then
        log_error "Python 3 is required but not found"
        log_error "Please install Python 3 from https://www.python.org/"
        exit 1
    fi
    log_info "Python 3: $(python3 --version)"
}

check_homebrew() {
    if ! command -v brew >/dev/null 2>&1; then
        log_warning "Homebrew not found. Installing Homebrew..."
        printf "\n"
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

        # Add Homebrew to PATH for Apple Silicon Macs
        if [ -f /opt/homebrew/bin/brew ]; then
            eval "$(/opt/homebrew/bin/brew shellenv)"
        fi

        log_success "Homebrew installed successfully"
    else
        log_info "Homebrew: $(brew --version | head -n 1)"
    fi
}

check_uv() {
    if ! command -v uv >/dev/null 2>&1; then
        log_warning "uv not found. Installing uv..."
        printf "\n"
        curl -LsSf https://astral.sh/uv/install.sh | sh

        # Source the shell config to get uv in PATH
        if [ -f "$HOME/.cargo/env" ]; then
            . "$HOME/.cargo/env"
        fi

        log_success "uv installed successfully"
    else
        log_info "uv: $(uv --version)"
    fi
}

check_mkvtoolnix() {
    if ! command -v mkvmerge >/dev/null 2>&1 || ! command -v mkvextract >/dev/null 2>&1; then
        log_warning "mkvtoolnix not found. Installing via Homebrew..."
        printf "\n"

        # Ensure Homebrew is available first
        check_homebrew

        brew install mkvtoolnix
        log_success "mkvtoolnix installed successfully"
    else
        log_info "mkvmerge: $(mkvmerge --version | head -n 1)"
    fi
}

check_dependencies() {
    log_info "Checking dependencies..."
    printf "\n"

    # Check Python (required, must be pre-installed)
    check_python

    # Check and install uv if needed
    check_uv

    # Check and install mkvtoolnix if needed
    check_mkvtoolnix

    printf "\n"
    log_success "All dependencies satisfied"
    printf "\n"
}

# Function to print usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS] <directory>

Translate movie/anime subtitles from English to Polish in MKV files.
Processes each file completely before moving to the next (extract → translate → apply).

ARGUMENTS:
  directory         Directory containing MKV files

OPTIONS:
  --device DEVICE   Translation device: auto, cpu, cuda, mps (default: auto)
  --batch-size N    Subtitle lines per batch (default: 16)
  --backup          Create .bak backup of original MKV files
  --keep-srt        Keep intermediate SRT files (default: delete after processing)
  -h, --help        Show this help message

COMPATIBILITY:
  - POSIX-compliant (works with sh, bash, zsh)
  - macOS support (auto-installs dependencies)
  - Requires: Python 3 (pre-installed on macOS)

AUTO-INSTALLATION (macOS):
  - Homebrew (if needed)
  - uv (Python package manager)
  - mkvtoolnix (for MKV manipulation)

  Everything is installed automatically on first run!

BENEFITS:
  - Zero manual setup - just run the script
  - Process files one-at-a-time: Episode 1 is ready while Episode 2 is processing
  - Resume-friendly: Skip already processed files automatically
  - Memory efficient: Loads translation model once per file

EXAMPLES:
  # Full workflow with defaults
  $0 /path/to/anime

  # With backup and keep SRT files
  $0 --backup --keep-srt /path/to/movies

  # Custom translation settings
  $0 --device mps --batch-size 32 /path/to/anime

EOF
    exit 0
}

# Parse arguments
DIRECTORY=""
DEVICE="auto"
BATCH_SIZE="16"
BACKUP=""
KEEP_SRT="false"

while [ $# -gt 0 ]; do
    case "$1" in
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --backup)
            BACKUP="--backup"
            shift
            ;;
        --keep-srt)
            KEEP_SRT="true"
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            if [ -z "$DIRECTORY" ]; then
                DIRECTORY="$1"
            else
                log_error "Unknown option: $1"
                exit 1
            fi
            shift
            ;;
    esac
done

# Validate directory argument
if [ -z "$DIRECTORY" ]; then
    log_error "Directory argument is required"
    usage
fi

if [ ! -d "$DIRECTORY" ]; then
    log_error "Directory not found: $DIRECTORY"
    exit 1
fi

# Convert to absolute path
DIRECTORY=$(cd "$DIRECTORY" && pwd)

# Check and install dependencies if needed (macOS only for now)
printf "\n"
printf "==========================================\n"
printf "  Dependency Check\n"
printf "==========================================\n"
printf "\n"
check_dependencies

# Print configuration
printf "\n"
printf "==========================================\n"
printf "  Movie Translator - Sequential Workflow\n"
printf "==========================================\n"
printf "\n"
log_info "Directory: $DIRECTORY"
log_info "Device: $DEVICE"
log_info "Batch Size: $BATCH_SIZE"
log_info "Backup: ${BACKUP:-disabled}"
log_info "Keep SRT files: $KEEP_SRT"
printf "\n"

# Count MKV files first
TOTAL_FILES=$(find "$DIRECTORY" -maxdepth 1 -name "*.mkv" -type f | wc -l | tr -d ' ')

if [ "$TOTAL_FILES" -eq 0 ]; then
    log_warning "No MKV files found in directory"
    exit 0
fi

log_info "Found $TOTAL_FILES MKV file(s)"
printf "\n"

# Counters
COMPLETED=0
SKIPPED=0
FAILED=0

# Process each file sequentially (using portable approach)
find "$DIRECTORY" -maxdepth 1 -name "*.mkv" -type f | sort | while read -r MKV_FILE; do
    FILENAME=$(basename "$MKV_FILE")
    FILESTEM="${FILENAME%.mkv}"
    CURRENT=$((COMPLETED + SKIPPED + FAILED + 1))

    printf "\n"
    printf "==========================================\n"
    log_progress "${BOLD}[$CURRENT/$TOTAL_FILES] Processing: $FILENAME${NC}"
    printf "==========================================\n"
    printf "\n"

    PL_SRT="${DIRECTORY}/${FILESTEM}_pl.srt"

    # Step 1: Extract English subtitles
    log_info "Step 1/3: Extracting English subtitles..."
    if uv run srt-extract "$MKV_FILE"; then
        log_success "Extraction complete"
    else
        log_error "Extraction failed for $FILENAME"
        FAILED=$((FAILED + 1))
        continue
    fi

    # Find the extracted English subtitle (could be .ass, .ssa, or .srt)
    EN_SRT=""
    for ext in .ass .ssa .srt; do
        candidate="${DIRECTORY}/${FILESTEM}_en${ext}"
        if [ -f "$candidate" ]; then
            EN_SRT="$candidate"
            log_info "Found extracted subtitle: $(basename "$EN_SRT")"
            break
        fi
    done

    if [ -z "$EN_SRT" ]; then
        log_error "Could not find extracted English subtitle for $FILENAME"
        FAILED=$((FAILED + 1))
        continue
    fi

    printf "\n"

    # Step 2: Translate to Polish (always output as .srt)
    log_info "Step 2/3: Translating to Polish..."
    if uv run srt-translate "$EN_SRT" "$PL_SRT" --device "$DEVICE" --batch-size "$BATCH_SIZE"; then
        log_success "Translation complete"
    else
        log_error "Translation failed for $FILENAME"
        FAILED=$((FAILED + 1))
        continue
    fi

    printf "\n"

    # Step 3: Apply subtitles to MKV
    log_info "Step 3/3: Applying subtitles to MKV..."
    if uv run srt-apply "$MKV_FILE" $BACKUP; then
        log_success "Subtitles applied successfully"
        COMPLETED=$((COMPLETED + 1))
    else
        log_error "Apply failed for $FILENAME"
        FAILED=$((FAILED + 1))
        continue
    fi

    # Clean up SRT files unless --keep-srt flag is set
    if [ "$KEEP_SRT" = "false" ]; then
        log_info "Cleaning up SRT files..."
        rm -f "$EN_SRT" "$PL_SRT"
    fi

    printf "\n"
    log_success "✓ $FILENAME is ready to watch!"
    log_progress "Progress: $COMPLETED/$TOTAL_FILES files completed"
done

# Note: Counters don't persist outside the while loop in POSIX sh
# So we recount at the end
printf "\n"
printf "==========================================\n"
printf "  🎉 Workflow Complete!\n"
printf "==========================================\n"
printf "\n"
log_success "All files processed!"
printf "\n"
log_info "All processed MKV files now have:"
printf "  ✅ English subtitle (original, default)\n"
printf "  ✅ Polish subtitle (AI-generated)\n"
printf "\n"
log_success "Done! 🎬"
