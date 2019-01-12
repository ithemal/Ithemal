#!/usr/bin/env bash

SAVEDIR="${ITHEMAL_HOME}/learning/pytorch/saved"

if [ ! -f "${SAVEDIR}/time_skylake_1217.data" ]; then
   mkdir -p "${SAVEDIR}"
   pushd "${SAVEDIR}"
   wget https://www.dropbox.com/s/qjcjje5hjrljd5a/time_skylake_1217.data
   popd
fi
