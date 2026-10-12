#!/usr/bin/env bash
# Run the commands of a chapter's README.md and show each command with its output.
#
# Usage:
#   ./examples.sh          # run every command, in README order
#   ./examples.sh 7        # run command [7] only (several numbers are fine)
#   ./examples.sh -l       # list the numbered commands without running them
#
# The commands are read from README.md itself: every ```bash block, in order,
# except the build block (its first command after cd is cmake) and file excerpts (a ***label*** line above
# the fence). The README marks each block's output with the same number, e.g.
# "Real output [7]:". Build the chapter first (standalone, see the top-level README).
set -uo pipefail
# This file lives in toy/; each chapter has a symlink ChN/examples.sh -> ../examples.sh.
# $0 is the path it was invoked by, so cd'ing to its directory selects the chapter.
cd "$(dirname "$0")"
if [[ ! $(basename "$PWD") =~ ^Ch[0-9]+$ ]]; then
  echo "error: run it through a chapter's symlink, e.g. Ch2/examples.sh — it reads the README.md next to the symlink it is run through" >&2
  exit 2
fi

README=README.md

# Emit one record per runnable block: number, section, then the block text,
# separated by \x1e (fields) and \x1f (records).
extract() {
  awk '
    inb {
      if ($0 ~ /^ *``` *$/) {
        inb = 0
        first = body; sub(/^cd [^\n]*\n/, "", first)
        if (first ~ /^cmake /) skip = 1
        if (!skip) printf "%d\x1e%s\x1e%s\x1f", ++n, sec, body
        prev = $0
      } else {
        body = body substr($0, indent + 1) "\n"
      }
      next
    }
    fence { if ($0 ~ /^ *``` *$/) fence = 0; prev = $0; next }
    /^ *```bash *$/ {
      match($0, /^ */); indent = RLENGTH
      inb = 1; body = ""; skip = (prev ~ /^\*\*\*.*\*\*\*$/); next
    }
    /^ *```/ { fence = 1; prev = $0; next }
    /^#{2,3} ([0-9]+\.|[A-F]\.[0-9]|Appendix )/ { sec = $0; sub(/^#+ /, "", sec) }
    /[^ ]/ { prev = $0 }
  ' "$README"
}

if [[ -t 1 ]]; then B=$'\e[1m'; D=$'\e[2m'; R=$'\e[0m'; else B=; D=; R=; fi

nums=() sects=() bodies=()
while IFS=$'\x1e' read -r -d $'\x1f' num sec body; do
  nums+=("$num"); sects+=("$sec"); bodies+=("$body")
done < <(extract)
total=${#nums[@]}

run_one() {
  local i=$(($1 - 1))
  printf '%s=== [%s] %s ===%s\n' "$B" "${nums[$i]}" "${sects[$i]}" "$R"
  printf '%s%s%s' "$D" "${bodies[$i]}" "$R"
  printf '%s--- output ---%s\n' "$B" "$R"
  bash -c "${bodies[$i]}" 2>&1
  local status=$?
  printf '%s--- exit status %d ---%s\n\n' "$B" "$status" "$R"
}

if [[ ${1:-} == -l || ${1:-} == --list ]]; then
  for ((i = 0; i < total; i++)); do
    first=$(grep -v '^cd ' <<<"${bodies[$i]}" | head -1)
    id=${sects[$i]%% *}
    [[ $id == Appendix ]] && id=${sects[$i]:9:1}
    id=${id%.}
    printf '[%2s] %-5s %s\n' "${nums[$i]}" "$id" "$first"
  done
  exit 0
fi

if [[ $# -eq 0 ]]; then
  for ((n = 1; n <= total; n++)); do run_one "$n"; done
  exit 0
fi

for n in "$@"; do
  if ! [[ $n =~ ^[0-9]+$ ]] || ((n < 1 || n > total)); then
    echo "error: no command [$n] — README.md has [1]..[$total] (see ./examples.sh -l)" >&2
    exit 2
  fi
  run_one "$n"
done
