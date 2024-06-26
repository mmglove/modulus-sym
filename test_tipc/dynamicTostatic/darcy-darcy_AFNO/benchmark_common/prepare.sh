if [ ! -f './modulus_sym_paddle_req_whl.tar.gz' ]; then
    wget https://paddle-qa.bj.bcebos.com/benchmark/pretrained/modulus_sym_paddle_req_whl.tar.gz
    tar xvf modulus_sym_paddle_req_whl.tar.gz
fi
pip install modulus_sym_paddle_req_whl/*
pip install -e .
pip list

if [ ! -f './examples_sym.zip' ]; then
    wget https://paddle-org.bj.bcebos.com/paddlescience/datasets/modulus/examples_sym.zip
fi

if [ ! -d './examples_sym' ]; then
    unzip examples_sym.zip
fi
unalias cp 2>/dev/null
\cp -r -f -v ./examples_sym/examples/* ./examples/

if [ ! -d './examples/darcy/datasets' ]; then
    mkdir -p ./examples/darcy/datasets && cd ./examples/darcy/datasets
    wget https://paddle-qa.bj.bcebos.com/benchmark/pretrained/Darcy_241.tar.gz
    tar xf Darcy_241.tar.gz
    cd -
fi
