cur=$(pwd)
cd zsim

export PINPATH=/data/scratch/charithm/libraries/install/pin-2.14
export LIBRARY_PATH=$LIBRARY_PATH:/data/scratch/charithm/libraries/install/libconfig-1.4.10/lib
export CPATH=$CPATH:/data/scratch/charithm/libraries/install/libconfig-1.4.10/include

export LIBRARY_PATH=$LIBRARY_PATH:/data/scratch/charithm/libraries/install/hdf5-1.10.2/lib
export CPATH=$CPATH:/data/scratch/charithm/libraries/install/hdf5-1.10.2/include

export LIBRARY_PATH=$LIBRARY_PATH:/data/scratch/charithm/libraries/install/libelf-0.8.13/lib
export CPATH=$CPATH:/data/scratch/charithm/libraries/install/libelf-0.8.13/include

scons -j24

cd $cur
