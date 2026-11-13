Here is the English translation of the provided text:

# Two-Stage Training and Testing Commands

The following commands are executed in the project root directory `/home/zzx/laji/A2S-v3`, using GPU 0, for the RGB task and the DUT-O test set.

## Stage 1: A2S + DINO affinity prior loss

### Training

```bash
python3 train.py a2s \
  --gpus=0 \
  --stage=1 \
  --trset=c \
  --vals=ce \
  --stage1_dino_weight=2.5
```

The training loss will be written to `./runs/a2s/resnet/base`. Open another terminal and start TensorBoard:

```bash
tensorboard --logdir ./runs --port 6006
```

Access `http://localhost:6006` in your browser to view `stage1/total`, `stage1/csd`, `stage1/btm`, `stage1/mse`, `stage1/dino_raw`, and `stage1/dino_weighted`.

Stage 1 trains for 20 epochs by default. The DINO affinity prior loss takes effect starting from the 2nd epoch. Offline tokens are read from the following directory:

```text
/home/zzx/sod/VRF_WSOD/TokenCut/DUTS-TR_train_token_similarity_40_40_384
```

The final weights are saved by default as:

```text
./weight/a2s/resnet/c/base/a2s_base_20.pth
```

### Testing

```bash
python3 test.py a2s \
  --gpus=0 \
  --stage=1 \
  --vals=ce \
  --weight=./weight/a2s/resnet/c/base/a2s_base_20.pth \
  --save
```

A2S tokens are only used for the training loss and are not read during testing. Prediction results are saved by default to:

```text
./result/a2s/resnet/base/c_DUT-O/final
```

## Generating Pseudo-labels Required for Stage 2

First, run A2S on DUTS-TR and use CRF to save pseudo-labels:

```bash
python3 test.py a2s \
  --gpus=0 \
  --stage=1 \
  --vals=cr \
  --weight=./weight/a2s/resnet/c/base/a2s_base_20.pth \
  --save \
  --crf
```

Place the generated results into the location required by the Stage 2 data loader:

```bash
mkdir -p ./pseudo/c/DUTS-TR
cp -a ./result/a2s/resnet/base/c_DUTS-TR/crf/. ./pseudo/c/DUTS-TR/
```

## Stage 2: CORNet + DINO token fusion

### Training

```bash
python3 train.py cornet \
  --gpus=0 \
  --stage=2 \
  --trset=c \
  --vals=ce \
  --olr
```

CORNet automatically reads and fuses offline tokens during training, so there is no need to additionally specify `use_offline_token`. The final weights are saved by default as:

```text
./weight/cornet/resnet/c/base/cornet_base_1.pth
```

To run a control experiment with RGB + token that does not use Depth/T, add `--only_rgb`. This mode will not read second modality files, nor will it create or execute the auxiliary modality encoder, Modal Decoder, and dual-modal hard selection path:

```bash
python3 train.py cornet \
  --gpus=0 \
  --stage=2 \
  --trset=t \
  --vals=te \
  --only_rgb \
  --sub=rgb_only_t
```

When testing RGB-only weights, `--only_rgb` must also be included.

The dual-modal CORNet uses mutually independent RGB/Modal Decoders. The DINO token only refines the RGB Decoder output. The final prediction compares the prediction confidence of the RGB/Modal branches pixel by pixel and directly adopts the logit of the branch with higher confidence. The training loss defaults to:

```text
L = L_fused + 0.2 * L_rgb + 0.2 * L_modal
```

The weights of the two auxiliary losses can be adjusted via `--rgb_aux_weight` and `--modal_aux_weight`. In TensorBoard, `fusion/modal_selection_ratio` represents the proportion of pixels where the Modal branch is adopted, and `fusion/modal_selection_white_is_modal` is the corresponding selection map.

### Joint Training of RGB, RGBD, and RGBT

During joint training, RGBD and RGBT share the same Modal Encoder/Decoder, and pure RGB samples skip the Modal branch. Each epoch is randomly shuffled normally and traverses all three datasets once, without repeating samples from RGBD/RGBT:

```bash
CUDA_VISIBLE_DEVICES=1 python3 train.py cornet \
  --gpus=0 \
  --stage=2 \
  --trset=cdt \
  --vals=ce,te \
  --sub=joint_cdt_last \
  --olr
```

### Testing

```bash
python3 test.py cornet \
  --gpus=0 \
  --stage=2 \
  --vals=ce \
  --weight=./weight/cornet/resnet/c/base/cornet_base_10.pth \
  --save
```

Prediction results are saved by default to:

```text
./result/cornet/resnet/base/c_DUT-O/final
```