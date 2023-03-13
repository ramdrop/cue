# CUE-Segmentation

###  Environment (Docker)
- Ubuntu 18.04
- PyTorch 1.10 + CUDA 11.1
- MinkowskiEngine 0.5.4

```shell
# Modify `TORCH_CUDA_ARCH_LIST` in docker/Dockfile to match your GPU, then run:

$local: docker build -t cue:1.0 docker
$local: docker run --gpus all --rm -itd --name cue -v /local_dir:/container_dir --shm-size 16G --ipc=host cue:1.0
$container: conda init
$(base)container: cd cue_feature/
$(base)container: ./install_env.sh

# Then modify `/opt/conda/envs/fpt/lib/python3.8/site-packages/torch/utils/tensorboard/__init__.py` by:
# removing: LooseVersion = distutils.version.LooseVersion
# adding: from distutils.version import LooseVersion
```


### Dataset
- Download ScanNetV2 dataset and preprocess it:

    ```shell
    ./download_scannet.sh
    python src/data/preprocess_scannet.py 
    ```

### Training and Evaluation
-  Mink

    ```shell
    python train.py --config=config/scannet/train_res16unet34c.gin
    python eval.py  --config=config/scannet/eval_res16unet34c.gin --ckpt_path=xxx.ckpt
    ```

- Mink+CUE
    ```shell
    python train.py --config=config/scannet/train_res16unet34c_prob.gin  --gpus=0
    python eval.py --config=config/scannet/eval_res16unet34c_prob.gin --ckpt_path=xxx.ckpt
    ```

- Mink+CUE+:
    ```
    python  train.py  --config=config/scannet/train_res16unet34c_probmg.gin  --gpus=2
    python  eval.py --config=config/scannet/eval_res16unet34c_probmg.gin --ckpt_path=xxx.ckpt
    ```

- Mink+SE:
    ```
    python eval.py  --config=config/scannet/eval_res16unet34c.gin --ckpt_path=xxx.ckpt
    ```
- Mink+AU (Aleatoric Uncertainty)
    ```
    python  train.py  --config=config/scannet/train_res16unet34c_aleatoric.gin  --gpus=2
    python  eval.py  --config=config/scannet/eval_res16unet34c_aleatoric.gin  --ckpt_path=xxx.ckpt
    ```

 - Mink+MCD (MC Dropout Uncertainty)
    ```
    python train.py  --config=config/scannet/train_res16unet34c_mc.gin  --gpus=1
    python eval.py --config=config/scannet/eval_res16unet34c_mc.gin --ckpt_path=xxx.ckpt
    ```
- Mink+DUL
    ```
    python train.py  --config=config/scannet/train_res16unet34c_dul.gin  --gpus=0
    python eval.py --config=config/scannet/eval_res16unet34c_dul.gin --ckpt_path=xxx.ckpt
    ```
- Mink+RUL
    ```
    python train.py  --config=config/scannet/train_res16unet34c_rul.gin  --gpus=2
    python eval.py --config=config/scannet/eval_res16unet34c_rul.gin --ckpt_path=xxx.ckpt
    ```
### Quantative and Qualitative Visualization

First, complete the eval.py script. Then, populate the variables in src/mbox/com.py. Follow the steps below for quantitative and qualitative visualization.

- To calculate Expected Calibration Error (ECE), run `python src/mbox/qn_sigma.py --uncertainty_method=[method]`. The `ece` folder will be created in the [log_method] directory.
- To visualize ECE, populate the variables and run `python src/mbox/plot_ece_break.py`. The `meta` folder will be created in the [log_method] directory.
- To view point cloud visualization, run `python src/mbox/qa_sigma.py --uncertainty_method=[method]`.