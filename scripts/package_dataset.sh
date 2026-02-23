#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR="."
OUTPUT_DIR="."
ARCHIVE_NAME="preprocessed_npy"
FORMAT="zip"

usage() {
  cat <<'EOF'
Usage: package_dataset.sh [options]

Options:
  -s, --source DIR     Source preprocessed NPY directory (default: .)
  -o, --output DIR     Output directory for the archive (default: .)
  -n, --name NAME      Base archive name without extension (default: preprocessed_npy)
  -f, --format FORMAT  Archive format: tar.gz or zip (default: zip)
  -h, --help           Show this help and exit

Examples:
  ./scripts/package_dataset.sh
  ./scripts/package_dataset.sh -s ./preprocessed -o /tmp -n preprocessed_npy
  ./scripts/package_dataset.sh -f tar.gz
EOF
}

humanize_bytes() {
  local bytes="$1"
  if command -v numfmt >/dev/null 2>&1; then
    numfmt --to=iec --suffix=B "$bytes"
  else
    printf "%s bytes" "$bytes"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -s|--source)
      SOURCE_DIR="$2"
      shift 2
      ;;
    -o|--output)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    -n|--name)
      ARCHIVE_NAME="$2"
      shift 2
      ;;
    -f|--format)
      FORMAT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
 done

if [[ ! -d "$SOURCE_DIR" ]]; then
  echo "Source directory not found: $SOURCE_DIR" >&2
  exit 1
fi

if [[ ! -d "$OUTPUT_DIR" ]]; then
  echo "Output directory not found: $OUTPUT_DIR" >&2
  exit 1
fi

DATASET_BYTES=$(du -sb "$SOURCE_DIR" | awk '{print $1}')
FREE_BYTES=$(df -B1 "$OUTPUT_DIR" | awk 'NR==2 {print $4}')
REQUIRED_HUMAN=$(humanize_bytes "$DATASET_BYTES")
FREE_HUMAN=$(humanize_bytes "$FREE_BYTES")

echo "Dataset size: $REQUIRED_HUMAN"
echo "Free space:   $FREE_HUMAN"

if [[ "$FREE_BYTES" -lt "$DATASET_BYTES" ]]; then
  echo "Not enough free space in $OUTPUT_DIR" >&2
  exit 1
fi

ARCHIVE_PATH="$(cd "$OUTPUT_DIR" && pwd)/${ARCHIVE_NAME}.$FORMAT"

case "$FORMAT" in
  tar.gz)
    tar -czf "$ARCHIVE_PATH" -C "$SOURCE_DIR" .
    ;;
  zip)
    if command -v pv >/dev/null 2>&1; then
      FILE_COUNT=$(find "$SOURCE_DIR" -type f | wc -l | awk '{print $1}')
      (cd "$SOURCE_DIR" && find . -type f -print0) \
        | pv -0 -l -s "$FILE_COUNT" -N "files" \
        | tr '\0' '\n' \
        | (cd "$SOURCE_DIR" && zip -q -@ "$ARCHIVE_PATH")
    else
      echo "pv not found; creating zip without progress bar."
      (cd "$SOURCE_DIR" && zip -r "$ARCHIVE_PATH" .)
    fi
    ;;
  *)
    echo "Unsupported format: $FORMAT" >&2
    exit 1
    ;;
 esac

if command -v sha256sum >/dev/null 2>&1; then
  sha256sum "$ARCHIVE_PATH" > "${ARCHIVE_PATH}.sha256"
  echo "SHA256: ${ARCHIVE_PATH}.sha256"
fi

echo "Archive created: $ARCHIVE_PATH"
