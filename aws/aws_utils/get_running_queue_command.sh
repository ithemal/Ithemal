#!/usr/bin/env bash

QUEUE_PROC=$(pgrep -f queue_process.py)
if [[ ! "${QUEUE_PROC}" ]]; then
    exit 0
fi

INVOKED_SHELL_PROC=$(pgrep -P ${QUEUE_PROC})
if [[ ! "${INVOKED_SHELL_PROC}" ]]; then
    exit 0
fi
RUNNING_COM=$(pgrep -P $INVOKED_SHELL_PROC)

if [[ ! "${RUNNING_COM}" ]]; then
    exit 0
fi

ps -q $RUNNING_COM -o "args="
