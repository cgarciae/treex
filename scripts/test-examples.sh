
set -e

#----------------------------------------------------------------
# test examples
#----------------------------------------------------------------
for file in $(find examples -name '*.py' -not -name 'noisy_linear.py') ; do
    cmd="python $file --epochs 2 --steps-per-epoch 1 --batch-size 3"
    echo RUNNING: $cmd
    DISPLAY="" $cmd > /dev/null
done