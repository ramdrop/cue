# CUE-Feature
###  Environment (Docker)
- Ubuntu 18.04
- PyTorch 1.12 + CUDA 11.3
- MinkowskiEngine 0.5.4

To set up the environment, modify `TORCH_CUDA_ARCH_LIST` in `docker/Dockfile` to match your GPU, and then run the following commands:
```shell
$local: docker build -t cue:1.0 docker
$local: docker run --gpus all --rm -itd --name cue -v /local_dir:/container_dir --shm-size 16G --ipc=host cue:1.0
$container: conda init
$(base)container: cd cue_feature/
$(base)container: ./install_env.sh

```

### Dataset
    
To download the required datasets, run the following scripts:

    ```shell
    ./scripts/download_3dmatch.sh dbs/
    ./scripts/download_3dmatch_testbench.sh dbs/
    ```

### Train CUE/CUE+

- First, to train FCGF , run the following command:
    ```shell
    python main.py  train=3dmatch_pair
    ```
- Then, to train CUE, populate weights in conf/train/3dmatch_pair_btl with the saved checkpoint path, for example: `weights: logs/HCL_0419_161400/best_val_checkpoint.pth`, and run the following command:
    ```shell
    python main.py train=3dmatch_pair_btl
    ```
- Alternatively, to train CUE+, populate weights in conf/train/3dmatch_pair_mbtl with the saved checkpoint path and run the following command:
    ```shell
    python main.py train=3dmatch_pair_mbtl
    ```

### Evaluate CUE/CUE+
- Eval FCGF: 
    ```shell
    python eval/eval_3dmatch.py  --model=[saved_checkpoint.pth] --extract_features=1    --evaluate_FMR=1
    ```
- Eval CUE: 
    ```shell
    python eval/eval_3dmatch.py  --model=[saved_checkpoint.pth] --evaluate_ECE=1
    ```    
- Eval CUE+: 
    ```shell
    python eval/eval_3dmatch.py  --model=[saved_checkpoint.pth] --evaluate_ECE=1
    ```        
### Plot ECE of CUE/CUE+
- To plot the ECE of CUE/CUE+, run the following commands:
    ```
    python eval/ece_rg.py
    # populate the ece_results.pickle path in eval/plot.ece.py and then run
    python eval/plot_ece.py
    ```