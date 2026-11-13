# 两阶段训练与测试命令

以下命令均在项目根目录 `/home/zzx/laji/A2S-v3` 下执行，使用 GPU 0、RGB 任务和 DUT-O 测试集。

## Stage 1：A2S + DINO affinity prior loss

### 训练

```bash
python3 train.py a2s \
  --gpus=0 \
  --stage=1 \
  --trset=c \
  --vals=ce \
  --stage1_dino_weight=2.5
```

训练 loss 会写入 `./runs/a2s/resnet/base`。另开一个终端启动 TensorBoard：

```bash
tensorboard --logdir ./runs --port 6006
```

浏览器访问 `http://localhost:6006`，可以查看 `stage1/total`、`stage1/csd`、`stage1/btm`、`stage1/mse`、`stage1/dino_raw` 和 `stage1/dino_weighted`。

Stage 1 默认训练 20 个 epoch。DINO affinity prior loss 从第 2 个 epoch 开始生效，离线 token 从以下目录读取：

```text
/home/zzx/sod/VRF_WSOD/TokenCut/DUTS-TR_train_token_similarity_40_40_384
```

最终权重默认保存为：

```text
./weight/a2s/resnet/c/base/a2s_base_20.pth
```

### 测试

```bash
python3 test.py a2s \
  --gpus=0 \
  --stage=1 \
  --vals=ce \
  --weight=./weight/a2s/resnet/c/base/a2s_base_20.pth \
  --save
```

A2S 的 token 只用于训练损失，测试时不读取 token。预测结果默认保存至：

```text
./result/a2s/resnet/base/c_DUT-O/final
```

## 生成 Stage 2 所需的伪标签

先在 DUTS-TR 上运行 A2S，并使用 CRF 保存伪标签：

```bash
python3 test.py a2s \
  --gpus=0 \
  --stage=1 \
  --vals=cr \
  --weight=./weight/a2s/resnet/c/base/a2s_base_20.pth \
  --save \
  --crf
```

将生成结果放到 Stage 2 数据加载器要求的位置：

```bash
mkdir -p ./pseudo/c/DUTS-TR
cp -a ./result/a2s/resnet/base/c_DUTS-TR/crf/. ./pseudo/c/DUTS-TR/
```

## Stage 2：CORNet + DINO token fusion

### 训练

```bash
python3 train.py cornet \
  --gpus=0 \
  --stage=2 \
  --trset=c \
  --vals=ce \
  --olr
```

CORNet 训练时会自动读取并融合离线 token，不需要额外指定 `use_offline_token`。最终权重默认保存为：

```text
./weight/cornet/resnet/c/base/cornet_base_1.pth
```

如需运行不使用 Depth/T 的 RGB + token 对照实验，增加
`--only_rgb`。该模式不会读取第二模态文件，也不会创建或执行辅助
模态编码器、Modal Decoder 和双模态硬选择路径：

```bash
python3 train.py cornet \
  --gpus=0 \
  --stage=2 \
  --trset=t \
  --vals=te \
  --only_rgb \
  --sub=rgb_only_t
```

测试 RGB-only 权重时也必须带上 `--only_rgb`。

双模态 CORNet 使用相互独立的 RGB/Modal Decoder。DINO token
只精炼 RGB Decoder 输出。最终预测逐像素比较 RGB/Modal 预测置信度，
并直接采用置信度更高的分支 logit。训练损失默认为：

```text
L = L_fused + 0.2 * L_rgb + 0.2 * L_modal
```

两个辅助损失权重可通过 `--rgb_aux_weight` 和
`--modal_aux_weight` 调整。TensorBoard 中的
`fusion/modal_selection_ratio` 表示 Modal 被采用的像素比例，
`fusion/modal_selection_white_is_modal` 为对应选择图。

### RGB、RGBD、RGBT 联合训练

联合训练时 RGBD 与 RGBT 共用同一个 Modal Encoder/Decoder，纯 RGB
样本跳过 Modal 分支。每个 epoch 普通随机打乱并遍历三套数据一次，
不对 RGBD/RGBT 进行重复采样：

```bash
CUDA_VISIBLE_DEVICES=1 python3 train.py cornet \
  --gpus=0 \
  --stage=2 \
  --trset=cdt \
  --vals=ce,te \
  --sub=joint_cdt_last \
  --olr
```

### 测试

```bash
python3 test.py cornet \
  --gpus=0 \
  --stage=2 \
  --vals=ce \
  --weight=./weight/cornet/resnet/c/base/cornet_base_10.pth \
  --save
```

预测结果默认保存至：

```text
./result/cornet/resnet/base/c_DUT-O/final
```
