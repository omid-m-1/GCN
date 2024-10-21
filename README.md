# 2 layer GCN with CUDA Kernel - Assignment 3

## Usage

To train GCN model run: `python main.py --dataset dataset` command. results for `cora, citeseer, pubmed and reddit` datasets are saved in results folder.
 
For compiling the kernel, enter the following command in the `deep-codegen` directory:
```bash
mkdir build && cd build
cmake ..
make -j
cp graphpy.cpython-38-x86_64-linux-gnu.so ../
```
