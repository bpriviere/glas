# ORCA baseline

Sample code based on the RVO2 library (https://github.com/snape/RVO2).

## Compile

```
git submodule init
git submodule update
mkdir build
cd build
cmake ..
make
```

## Run

(from the build directory)
```
./orca
```

## Visualize

(from the build directory)
```
python3 ../orca.py [--animate]
```
