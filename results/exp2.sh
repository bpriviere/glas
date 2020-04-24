ulimit -n 2048
rm -rf singleintegrator/exp2Empty*
rm -rf singleintegrator/exp2Barrier*
python3 singleintegrator/exp2.py --train
python3 singleintegrator/exp2.py --sim
python3 singleintegrator/exp2.py --plot
