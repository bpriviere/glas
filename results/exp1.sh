ulimit -n 2048
rm -rf singleintegrator/exp1Empty*
rm -rf singleintegrator/exp1Barrier*
python3 singleintegrator/exp1.py --train
python3 singleintegrator/exp1.py --sim
python3 singleintegrator/exp1.py --plot
