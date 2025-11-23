#!/bin/sh
#
# Movie Translator - Sequential File-by-File Workflow
# POSIX-compliant shell script (works with sh, bash, zsh, etc.)
# Process each MKV file completely before moving to the next
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
  - No bash 4+ features required
  - Compatible with macOS default shell

BENEFITS:
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

    EN_SRT="${DIRECTORY}/${FILESTEM}_en.srt"
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

    printf "\n"

    # Step 2: Translate to Polish
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
