export XLA_FLAGS=--xla_dump_to=/tmp/foo
rm -rf /tmp/foo
python benchmark/obsa_grad.py --geometry o

# assumes hloviz binary is at the root directory
# cp /tmp/foo/*4c*after*optimizations.txt 4c.txt
# ./hloviz --hlo 4c.txt --html 4c.html --raw-custom-call

# cp /tmp/foo/*mask*after*optimizations.txt mask.txt
# ./hloviz --hlo mask.txt --html mask.html --raw-custom-call

cp /tmp/foo/*eri*after*optimizations.txt latest.txt
./hloviz --hlo latest.txt --html latest.html --raw-custom-call
