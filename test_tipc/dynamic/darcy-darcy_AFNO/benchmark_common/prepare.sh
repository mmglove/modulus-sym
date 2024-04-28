pip install -e .

if [ ! -f './examples_sym.zip' ]; then
    wget https://paddle-org.bj.bcebos.com/paddlescience/datasets/modulus/examples_sym.zip
fi

if [ ! -d './examples_sym' ]; then
    unzip examples_sym.zip
    \cp -r -f -v ./examples_sym/examples/* ./examples/
fi

if [ ! -d './examples/darcy/datasets' ]; then
    mkdir -p ./examples/darcy/datasets && cd ./examples/darcy/datasets
    wget wget https://paddle-qa.bj.bcebos.com/benchmark/pretrained/Darcy_241.tar.gz
    tar xf Darcy_241.tar.gz
    cd -
fi
