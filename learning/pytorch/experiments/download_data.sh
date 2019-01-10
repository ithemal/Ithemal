#!/usr/bin/env bash

if [ ! -f "${ITHEMAL_HOME}/saved/time_skylake_1217.data" ]; then
   mkdir -p "${ITHEMAL_HOME}/saved"
   pushd "${ITHEMAL_HOME}/saved"
   wget https://www.dropbox.com/s/qjcjje5hjrljd5a/time_skylake_1217.data
   popd
fi
