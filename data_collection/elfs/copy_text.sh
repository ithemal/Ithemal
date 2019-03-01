#!/bin/bash


find $1 -print0 | while IFS= read -r -d '' filename
do 
    if file $filename | grep -q -i 'elf 64'; then
	output=$(echo $filename | sed -e 's/.*\/\(.*\)$/\1/')
	echo $output.o
	objcopy -O binary --only-section=.text $filename $output.o 
    fi
done
