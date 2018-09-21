cur=$(pwd)

#download Agner Fog's measurement tools
if [ -d "agner/testp" ]; then
    echo "Agner Fog's timing scripts are already downloaded"
else
    echo "downloading Agner Fog's timing scripts...."
    cd agner
    wget https://www.agner.org/optimize/testp.zip
    unzip -d testp testp.zip
fi

cd $cur
cd agner
#copy files
cp a64-out.sh testp/PMCTest
cp PMCTestB64.nasm testp/PMCTest
cd $cur

#give instructions to download IACA
if [ -f "iaca/iacaMarks.h" ] && [ -f "iaca/iaca" ]; then
    echo "IACA is already downloaded"
else
    echo "please download iaca and place iaca binary and iacaMarks.h in the iaca folder"
    echo "https://software.intel.com/en-us/articles/intel-architecture-code-analyzer"
fi

cd $cur

#download and build LLVM if necessary
if [ ! -d "llvm" ]; then
    echo "Downloading LLVM..."
    git clone https://github.com/llvm-mirror/llvm.git llvm
else
    echo "LLVM already downloaded..."
fi

cd llvm
git pull
cd $cur

if [ ! -d "llvm-build/bin/llvm-mca" ]; then
    echo "Building llvm-mca"
    mkdir -p llvm-build
    cd llvm-build
    cmake -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD=X86 ../llvm
    make -j24 llvm-mca
else
    echo "llvm-mca already built"
fi

cd $cur
