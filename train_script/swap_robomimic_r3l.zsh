#!/usr/bin/env zsh
set -euo pipefail

# Launch four CADR-U R3L Robomimic image runs in one tmux session.
# Each task gets one physical GPU and a PyTorch CUDA allocator cap.
#
# Usage:
#   zsh train_script/swap_robomimic_r3l.zsh
#   GPU_MEMORY_GB=25 TMUX_SESSION_NAME=my_r3l zsh train_script/swap_robomimic_r3l.zsh seed=0
#   DRY_RUN=1 zsh train_script/swap_robomimic_r3l.zsh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="${DICE_RL_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
RUNNER="${R3L_RUNNER:-${SCRIPT_DIR}/run_r3l_robomimic_post_training.zsh}"

SESSION_NAME="${TMUX_SESSION_NAME:-swap_robomimic_r3l}"
WANDB_PROJECT="${R3L_WANDB_PROJECT:-R3L_robomimic}"
GPU_MEMORY_GB="${GPU_MEMORY_GB:-25}"

typeset -a tasks gpus extra_overrides
tasks=(can square tool-hang transport)
gpus=(2 3 4 5)
extra_overrides=("$@")

if [[ "${DRY_RUN:-0}" != "1" ]] && ! command -v tmux >/dev/null 2>&1; then
  print -u2 "tmux not found. Install tmux or run with DRY_RUN=1 to inspect commands."
  exit 127
fi

if [[ ! -f "${RUNNER}" ]]; then
  print -u2 "Missing R3L runner: ${RUNNER}"
  exit 1
fi

if ((${#tasks[@]} != ${#gpus[@]})); then
  print -u2 "Internal error: task count and GPU count differ."
  exit 1
fi

build_window_command() {
  local task="$1"
  local gpu="$2"
  local cmd override

  cmd="cd ${(q)ROOT_DIR}; "
  cmd+="export CUDA_VISIBLE_DEVICES=${(q)gpu}; "
  cmd+="export EGL_DEVICE_ID=0; "
  cmd+="export MUJOCO_EGL_DEVICE_ID=0; "
  cmd+="export USE_CADR_U=1; "
  cmd+="export R3L_REMAP_CUDA_VISIBLE_DEVICES=0; "
  cmd+="export R3L_WANDB_PROJECT=${(q)WANDB_PROJECT}; "
  cmd+="export DICE_RL_CUDA_MEMORY_LIMIT_GB=${(q)GPU_MEMORY_GB}; "
  cmd+="zsh ${(q)RUNNER} ${(q)task} device=cuda:0"

  for override in "${extra_overrides[@]}"; do
    cmd+=" ${(q)override}"
  done

  print -r -- "${cmd}"
}

if [[ "${DRY_RUN:-0}" != "1" ]] && tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  if [[ "${KILL_EXISTING:-0}" == "1" ]]; then
    tmux kill-session -t "${SESSION_NAME}"
  else
    print -u2 "tmux session already exists: ${SESSION_NAME}"
    print -u2 "Use KILL_EXISTING=1 to replace it, or set TMUX_SESSION_NAME to another name."
    exit 1
  fi
fi

for idx in {1..${#tasks[@]}}; do
  task="${tasks[$idx]}"
  gpu="${gpus[$idx]}"
  window_name="${task}"
  window_cmd="$(build_window_command "${task}" "${gpu}")"

  print "[swap_robomimic_r3l] ${task}: GPU ${gpu}, CUDA memory cap ${GPU_MEMORY_GB} GiB, wandb project ${WANDB_PROJECT}"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    print -r -- "${window_cmd}"
    continue
  fi

  if ((idx == 1)); then
    tmux new-session -d -s "${SESSION_NAME}" -n "${window_name}" "${window_cmd}"
    tmux set-option -t "${SESSION_NAME}" remain-on-exit on >/dev/null
  else
    tmux new-window -t "${SESSION_NAME}:" -n "${window_name}" "${window_cmd}"
  fi
done

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

print "Started tmux session: ${SESSION_NAME}"
print "Attach with: tmux attach -t ${SESSION_NAME}"

if [[ "${ATTACH:-0}" == "1" ]]; then
  exec tmux attach -t "${SESSION_NAME}"
fi
