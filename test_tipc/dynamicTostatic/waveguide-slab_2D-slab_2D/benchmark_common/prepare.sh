pip install https://paddle-qa.bj.bcebos.com/benchmark/pretrained/sympy-1.12.1-py3-none-any.whl
pip install https://paddle-qa.bj.bcebos.com/benchmark/pretrained/mpmath-1.3.0-py3-none-any.whl

pip install -e .

if [ ! -f './examples_sym.zip' ]; then
    wget https://paddle-org.bj.bcebos.com/paddlescience/datasets/modulus/examples_sym.zip
fi

if [ ! -d './examples_sym' ]; then
    unzip examples_sym.zip
fi
unalias cp 2>/dev/null
\cp -r -f -v ./examples_sym/examples/* ./examples/
