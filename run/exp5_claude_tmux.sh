#!/usr/bin/env bash
# Keep the exp5 Claude Code conversation running on this server after VS Code / SSH is closed.
#
# A live VS Code chat cannot be moved.  This resumes the SAME conversation (by session id) in the
# Claude Code CLI inside a detached tmux session, which survives closing VS Code and SSH.  An
# optional initial prompt makes it continue working immediately, without anyone attaching.
#
#   bash run/exp5_claude_tmux.sh                      # start tmux session "exp5-claude"
#   PROMPT="continue" bash run/exp5_claude_tmux.sh    # ... and send it a first message
#   REMOTE_CONTROL=1 bash run/exp5_claude_tmux.sh     # also enable Remote Control (claude.ai / mobile)
#   tmux attach -t exp5-claude                        # watch or talk to it; Ctrl-b d detaches again
#   tmux kill-session -t exp5-claude                  # stop it (GPU jobs in their own tmux sessions keep running)
#
# Use ONE copy of the conversation at a time: stop typing in the VS Code panel once this runs.
set -euo pipefail
cd "$(dirname "$0")/.."
SESSION_ID=${SESSION_ID:-53f177e3-9b65-405e-b861-20fbeaad9509}
TMUX_NAME=${TMUX_NAME:-exp5-claude}
CLAUDE=${CLAUDE:-$(command -v claude 2>/dev/null || ls -d "$HOME"/.vscode-server/extensions/anthropic.claude-code-*/resources/native-binary/claude | sort -V | tail -1)}
[[ -x "$CLAUDE" ]] || { echo "Claude Code CLI not found (set CLAUDE=/path/to/claude)"; exit 1; }

args=(--resume "$SESSION_ID" --permission-mode auto --model 'opus[1m]' --effort max -n exp5)
[[ "${REMOTE_CONTROL:-0}" == 1 ]] && args+=(--remote-control exp5-gmmd)
[[ -n "${PROMPT:-}" ]] && args+=("$PROMPT")

if tmux has-session -t "$TMUX_NAME" 2>/dev/null; then
  echo "tmux session '$TMUX_NAME' already exists: tmux attach -t $TMUX_NAME"
  exit 0
fi
# `exec bash` keeps the window open after claude exits, so an error stays readable on attach
tmux new-session -d -s "$TMUX_NAME" -c "$PWD" -x 220 -y 50 "$(printf '%q ' "$CLAUDE" "${args[@]}"); exec bash"
echo "started '$TMUX_NAME' ($CLAUDE, session $SESSION_ID): tmux attach -t $TMUX_NAME"
