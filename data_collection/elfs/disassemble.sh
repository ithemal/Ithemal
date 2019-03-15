for filename in *.o; do
    echo $filename
    ../build/bin/disassembler $filename
done
