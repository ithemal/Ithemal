#!/bin/bash

objcopy -O binary --only-section=.text $1 $2
