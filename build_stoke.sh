export CLN_CFLAGS="-I$CLN/include/" 
export CLN_LIBS="-L$CLN/lib/ -lcln"
export BOOST_ROOT="/data/scratch/charithm/libraries/install/boost/boost_1_59_0"

export VPATH=/data/scratch/charithm/libraries/install/iml/include

#boost
export CPATH=$CPATH:/data/scratch/charithm/libraries/install/boost_built/include
export LIBRARY_PATH=$LIBRARY_PATH:/data/scratch/charithm/libraries/install/boost_built/lib

#iml
export CPATH=$CPATH:/data/scratch/charithm/libraries/install/iml/include
export LIBRARY_PATH=$LIBRARY_PATH:/data/scratch/charithm/libraries/install/iml/lib

#cln
export CPATH=$CPATH:/data/scratch/charithm/libraries/install/cln-1.3.4/include
export LIBRARY_PATH=$LIBRARY_PATH:/data/scratch/charithm/libraries/install/cln-1.3.4/lib

#gmp
export CPATH=$CPATH:/data/scratch/charithm/libraries/install/gmp-4.3.2/include
export LIBRARY_PATH=$LIBRARY_PATH:/data/scratch/charithm/libraries/install/gmp-4.3.2/lib

#cblas
export CPATH=$CPATH:/data/scratch/charithm/libraries/src/CBLAS/include
export LIBRARY_PATH=$LIBRARY_PATH:/data/scratch/charithm/libraries/src/CBLAS/lib

#gmp
export CPATH=$CPATH:/data/scratch/charithm/libraries/src/BLAS-3.8.0
export LIBRARY_PATH=$LIBRARY_PATH:/data/scratch/charithm/libraries/src/BLAS-3.8.0


cd stoke
make
cd ..
