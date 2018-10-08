#!/usr/bin/env bash

SESSION=$(tmux ls -F '#S #{session_attached}' | grep ' 0$' | head -n 1 | awk '{$NF=""; print $0}' | awk '{$1=$1;print}')

if [ ! -z "${SESSION}" ]; then
    tmux attach -t "${SESSION}"
else
    tmux new "bash -l"
fi
