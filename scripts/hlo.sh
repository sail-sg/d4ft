# Copyright 2023 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
