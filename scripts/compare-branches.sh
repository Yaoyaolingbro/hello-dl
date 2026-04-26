#!/usr/bin/env bash
set -euo pipefail

branch_a="codex"
branch_b="claude-code"
port_a="8010"
port_b="8011"
open_browser=1
serve_after_build=1

usage() {
  cat <<'USAGE'
Build and preview two MkDocs branches side by side.

Default behavior:
  scripts/compare-branches.sh

This builds:
  codex       -> http://127.0.0.1:8010
  claude-code -> http://127.0.0.1:8011

Each branch is compiled with:
  mkdocs build --strict

Options:
  --branch-a NAME   First branch to preview.  Default: codex
  --branch-b NAME   Second branch to preview. Default: claude-code
  --port-a PORT     First preview port.     Default: 8010
  --port-b PORT     Second preview port.    Default: 8011
  --no-open         Do not open browser tabs automatically.
  --build-only      Build both branches but do not start preview servers.
  -h, --help        Show this help message.

Output directories:
  .branch-previews/site/<branch>      Built static sites
  .worktrees/compare/<branch>         Git worktrees for branches not currently checked out

Press Ctrl-C to stop the preview servers.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --branch-a)
      branch_a="${2:?--branch-a requires a branch name}"
      shift 2
      ;;
    --branch-b)
      branch_b="${2:?--branch-b requires a branch name}"
      shift 2
      ;;
    --port-a)
      port_a="${2:?--port-a requires a port}"
      shift 2
      ;;
    --port-b)
      port_b="${2:?--port-b requires a port}"
      shift 2
      ;;
    --no-open)
      open_browser=0
      shift
      ;;
    --build-only)
      serve_after_build=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Run with --help for usage." >&2
      exit 2
      ;;
  esac
done

repo_root="$(git rev-parse --show-toplevel)"
preview_root="$repo_root/.branch-previews"
worktree_root="$repo_root/.worktrees/compare"

safe_name() {
  printf '%s' "$1" | tr '/ ' '__'
}

ensure_branch_exists() {
  local branch="$1"
  if ! git -C "$repo_root" rev-parse --verify --quiet "refs/heads/$branch" >/dev/null; then
    echo "Branch not found: $branch" >&2
    exit 1
  fi
}

find_worktree_for_branch() {
  local branch="$1"
  local line=""
  local path=""
  local wt_branch=""

  while IFS= read -r line; do
    case "$line" in
      worktree\ *)
        path="${line#worktree }"
        ;;
      branch\ refs/heads/*)
        wt_branch="${line#branch refs/heads/}"
        if [[ "$wt_branch" == "$branch" ]]; then
          printf '%s\n' "$path"
          return 0
        fi
        ;;
    esac
  done < <(git -C "$repo_root" worktree list --porcelain)

  return 1
}

prepare_worktree() {
  local branch="$1"
  local existing=""
  local target=""

  if existing="$(find_worktree_for_branch "$branch")"; then
    printf '%s\n' "$existing"
    return 0
  fi

  mkdir -p "$worktree_root"
  target="$worktree_root/$(safe_name "$branch")"

  if [[ -e "$target" ]]; then
    echo "Worktree path already exists but is not attached to $branch: $target" >&2
    echo "Remove it or choose a different branch name." >&2
    exit 1
  fi

  git -C "$repo_root" worktree add "$target" "$branch" >/dev/null
  printf '%s\n' "$target"
}

build_branch() {
  local branch="$1"
  local worktree="$2"
  local site_dir="$preview_root/site/$(safe_name "$branch")"

  rm -rf "$site_dir"
  mkdir -p "$site_dir"

  echo "Building $branch from $worktree" >&2
  (
    cd "$worktree"
    mkdocs build --strict --site-dir "$site_dir"
  )

  printf '%s\n' "$site_dir"
}

server_pids=()

stop_servers() {
  local pid=""
  for pid in "${server_pids[@]:-}"; do
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill "$pid" >/dev/null 2>&1 || true
    fi
  done
}

stop_and_exit() {
  stop_servers
  exit 0
}

start_server() {
  local label="$1"
  local site_dir="$2"
  local port="$3"
  local url="http://127.0.0.1:$port"

  python3 -m http.server "$port" --bind 127.0.0.1 --directory "$site_dir" >/dev/null 2>&1 &
  local pid=$!
  sleep 0.5

  if ! kill -0 "$pid" >/dev/null 2>&1; then
    echo "Failed to start server for $label on $url. Is the port already in use?" >&2
    exit 1
  fi

  server_pids+=("$pid")
  echo "$label: $url"

  if [[ "$open_browser" -eq 1 ]] && command -v open >/dev/null 2>&1; then
    open "$url" >/dev/null 2>&1 || true
  fi
}

if [[ "$branch_a" == "$branch_b" ]]; then
  echo "Branches must be different." >&2
  exit 1
fi

if [[ "$port_a" == "$port_b" ]]; then
  echo "Ports must be different." >&2
  exit 1
fi

ensure_branch_exists "$branch_a"
ensure_branch_exists "$branch_b"

worktree_a="$(prepare_worktree "$branch_a")"
worktree_b="$(prepare_worktree "$branch_b")"

site_a="$(build_branch "$branch_a" "$worktree_a")"
site_b="$(build_branch "$branch_b" "$worktree_b")"

if [[ "$serve_after_build" -eq 0 ]]; then
  echo "Built sites:"
  echo "  $branch_a: $site_a"
  echo "  $branch_b: $site_b"
  exit 0
fi

trap stop_servers EXIT
trap stop_and_exit INT TERM

echo
echo "Preview URLs:"
start_server "$branch_a" "$site_a" "$port_a"
start_server "$branch_b" "$site_b" "$port_b"
echo
echo "Press Ctrl-C to stop both preview servers."

while true; do
  sleep 1
done
