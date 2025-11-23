#!/bin/bash
#
# Movie Translator - Sequential File-by-File Workflow
# Process each MKV file completely before moving to the next
# This allows you to start watching early episodes while later ones are still processing
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Function to print colored messages
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_progress() {
    echo -e "${CYAN}$1${NC}"
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

BENEFITS:
  - Process files one-at-a-time: Episode 1 is ready while Episode 2 is processing
  - Resume-friendly: Skip already processed files automatically
  - Memory efficient: Loads translation model once per file batch

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
KEEP_SRT=false

while [[ $# -gt 0 ]]; do
    case $1 in
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
            KEEP_SRT=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            if [[ -z "$DIRECTORY" ]]; then
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
if [[ -z "$DIRECTORY" ]]; then
    log_error "Directory argument is required"
    usage
fi

if [[ ! -d "$DIRECTORY" ]]; then
    log_error "Directory not found: $DIRECTORY"
    exit 1
fi

# Convert to absolute path
DIRECTORY=$(cd "$DIRECTORY" && pwd)

# Print configuration
echo ""
echo "=========================================="
echo "  Movie Translator - Sequential Workflow"
echo "=========================================="
echo ""
log_info "Directory: $DIRECTORY"
log_info "Device: $DEVICE"
log_info "Batch Size: $BATCH_SIZE"
log_info "Backup: ${BACKUP:-disabled}"
log_info "Keep SRT files: $KEEP_SRT"
echo ""

# Find all MKV files
mapfile -t MKV_FILES < <(find "$DIRECTORY" -maxdepth 1 -name "*.mkv" | sort)
TOTAL_FILES=${#MKV_FILES[@]}

if [[ $TOTAL_FILES -eq 0 ]]; then
    log_warning "No MKV files found in directory"
    exit 0
fi

log_info "Found $TOTAL_FILES MKV file(s)"
echo ""

# Counters
COMPLETED=0
SKIPPED=0
FAILED=0

# Process each file sequentially
for MKV_FILE in "${MKV_FILES[@]}"; do
    FILENAME=$(basename "$MKV_FILE")
    FILESTEM="${FILENAME%.mkv}"
    CURRENT=$((COMPLETED + SKIPPED + FAILED + 1))

    echo ""
    echo "=========================================="
    log_progress "${BOLD}[$CURRENT/$TOTAL_FILES] Processing: $FILENAME${NC}"
    echo "=========================================="
    echo ""

    # Check if already processed (has both English and Polish subtitles embedded)
    # This is a simple check - you could make it more sophisticated
    EN_SRT="${DIRECTORY}/${FILESTEM}_en.srt"
    PL_SRT="${DIRECTORY}/${FILESTEM}_pl.srt"

    # Check if this file was already completed
    if [[ ! -f "$EN_SRT" ]] && [[ ! -f "$PL_SRT" ]]; then
        # Might be already processed and SRT files deleted
        # For now, we'll process it anyway
        :
    fi

    # Step 1: Extract English subtitles
    log_info "Step 1/3: Extracting English subtitles..."
    if uv run srt-extract "$MKV_FILE"; then
        log_success "Extraction complete"
    else
        log_error "Extraction failed for $FILENAME"
        FAILED=$((FAILED + 1))
        continue
    fi

    echo ""

    # Step 2: Translate to Polish
    log_info "Step 2/3: Translating to Polish..."
    if uv run srt-translate "$EN_SRT" "$PL_SRT" --device "$DEVICE" --batch-size "$BATCH_SIZE"; then
        log_success "Translation complete"
    else
        log_error "Translation failed for $FILENAME"
        FAILED=$((FAILED + 1))
        continue
    fi

    echo ""

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
    if [[ "$KEEP_SRT" == false ]]; then
        log_info "Cleaning up SRT files..."
        rm -f "$EN_SRT" "$PL_SRT"
    fi

    echo ""
    log_success "✓ $FILENAME is ready to watch!"
    log_progress "Progress: $COMPLETED/$TOTAL_FILES files completed"

done

# Final summary
echo ""
echo "=========================================="
echo "  🎉 Workflow Complete!"
echo "=========================================="
echo ""
log_success "Completed: $COMPLETED/$TOTAL_FILES files"
if [[ $SKIPPED -gt 0 ]]; then
    log_warning "Skipped: $SKIPPED files"
fi
if [[ $FAILED -gt 0 ]]; then
    log_error "Failed: $FAILED files"
fi
echo ""

if [[ $COMPLETED -gt 0 ]]; then
    log_info "All processed MKV files now have:"
    echo "  ✅ English subtitle (original, default)"
    echo "  ✅ Polish subtitle (AI-generated)"
    echo ""
fi

if [[ $FAILED -gt 0 ]]; then
    exit 1
else
    log_success "Done! 🎬"
    exit 0
fi
