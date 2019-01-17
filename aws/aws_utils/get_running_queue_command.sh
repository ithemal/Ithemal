#!/usr/bin/env bash

QUEUE_PROC=$(pgrep -f queue_process.py)
INVOKED_SHELL_PROC=$(pgrep -P ${QUEUE_PROC})
RUNNING_COM=$(pgrep -P $INVOKED_SHELL_PROC)

ps -q $RUNNING_COM -o "args="
