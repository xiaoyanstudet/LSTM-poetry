# LSTM 自动写诗项目代码改动记录

本文档记录本项目从师兄原始代码到当前版本的主要改动、改动原因、改动内容和验证结果。后续每次做较大代码调整时，建议继续按同样格式追加记录，并在 Git 中提交一个对应版本。

## 当前状态概览

- 当前入口文件：`main.py`
- 当前训练命令：`python main.py`
- 当前默认训练目标：七言绝句，`poem_type=7`
- 当前默认训练轮数：`main.py` 中 `--epochs` 默认值为 30
- 数据目录：`tangshi/`
- 训练产物目录：`data/`
- 最新 checkpoint：`data/checkpoints/poem_7_latest.pt`
- 指标图：`data/figures/training_metrics_7.png`
- 样例输出：`data/samples/`

## V0.0 师兄原始版本

### 为什么需要改

原始代码可以体现基本训练流程，但直接在当前 Mac 环境中运行会遇到几个问题：

- 项目目录没有自带 `tangshi/` 数据集。
- 代码依赖 `gensim.Word2Vec`，当前环境缺少 `gensim`。
- 代码写死 `cuda`，MacBook M 系列不能直接使用 CUDA。
- 没有断点续训机制，中断后需要从头训练。
- 没有系统化评测和可视化图表。
- 生成阶段自由采样，不能保证七言格式。

### 原始代码结构

- `main.py`：写死 `path1='tangshi'`、`path2='data'`，调用 `function.train(path1, path2, 7, 64)`。
- `function.py`：读取 JSON 文件，筛选五言/七言古诗，使用 Word2Vec 训练词向量，再训练 LSTM。
- `classmodel.py`：定义 Dataset 和 LSTM 模型，内部多处写死 `.to('cuda')`。

### 结果

原始版本在当前环境试跑时首先报错：

```text
ModuleNotFoundError: No module named 'gensim'
```

即使安装 `gensim`，后续也会因为缺少 `tangshi/` 数据集和 Mac 不支持 CUDA 而继续报错。

## V0.1 一键训练入口与 Mac MPS 适配

### 为什么这么改

目标是让项目能在 MacBook M 系列上直接运行，并尽量做到“一键开始训练”。

### 改动了什么

- 重写 `main.py`，改成命令行入口。
- 增加常用参数：
  - `--epochs`
  - `--batch-size`
  - `--poem-type`
  - `--device`
  - `--quick-test`
  - `--start-words`
- 增加设备自动选择逻辑：
  - 优先使用 `mps`
  - 其次使用 `cuda`
  - 最后回退到 `cpu`
- 设置 `PYTORCH_ENABLE_MPS_FALLBACK=1`，允许部分 MPS 不支持的算子回退执行。
- 新增 `start_training.command`，双击即可运行训练。

### 结果

现在可以直接运行：

```bash
python main.py
```

或双击：

```text
start_training.command
```

也可以先用快速测试检查流程：

```bash
python main.py --quick-test
```

## V0.2 模型结构与数据管线重构

### 为什么这么改

原始代码使用外部 `gensim.Word2Vec` 预训练字向量，增加了环境依赖，也让训练流程更复杂。对于字符级自动写诗实验，使用 PyTorch 自带 `nn.Embedding` 更直接，也更符合深度学习实验报告中的模型结构。

### 改动了什么

- 删除对 `gensim` 的依赖。
- `classmodel.py` 改为：
  - `nn.Embedding`
  - `nn.LSTM`
  - `nn.Dropout`
  - `nn.Linear`
- Dataset 改为直接返回：
  - 输入：当前字符序列
  - 标签：右移一位后的下一个字符序列
- `function.py` 中新增：
  - 构建词表
  - 诗句编码
  - 训练集/验证集划分

### 结果

模型不再依赖 `gensim`，环境更简单。当前 `requirements.txt` 只保留：

```text
torch
numpy
matplotlib
```

## V0.3 自动下载数据集

### 为什么这么改

原始代码要求手动准备 `tangshi/` 文件夹。为了实现“一键开始训练”，需要代码自动下载唐诗 JSON 数据。

### 改动了什么

- 在 `function.py` 中新增 `ensure_dataset()`。
- 默认下载 `chinese-poetry/chinese-poetry` 仓库中的全唐诗 JSON：
  - `poet.tang.0.json`
  - `poet.tang.1000.json`
  - ...
  - `poet.tang.57000.json`
- 下载目录为 `tangshi/`。
- 已存在且合法的 JSON 文件会自动跳过，不会重复下载。
- 新增 `--max-files`，可只下载前 N 个文件做测试。

### 结果

首次运行 `python main.py` 会自动下载数据。中途已经下载成功的 JSON 文件会保留，下次运行会从缺失文件继续。

## V0.4 断点续训与训练产物缓存

### 为什么这么改

完整训练可能耗时较长，中途关闭终端、断电或手动 `Ctrl+C` 都不应该导致训练进度丢失。

### 改动了什么

- 新增 checkpoint 保存：
  - `data/checkpoints/poem_7_latest.pt`
  - `data/checkpoints/poem_7_epoch_XXX.pt`
- checkpoint 内容包括：
  - 模型参数
  - 优化器状态
  - 当前 epoch
  - global step
  - 词表
  - 训练历史指标
  - 训练配置
- 默认开启自动续训。
- 新增 `--no-resume`，需要从头训练时可手动关闭续训。
- 捕获 `KeyboardInterrupt`，中断时保存最新 checkpoint。

### 结果

训练中断后，再次运行：

```bash
python main.py
```

会自动从 `data/checkpoints/poem_7_latest.pt` 继续训练。

## V0.5 评测函数与可视化

### 为什么这么改

自动写诗不能只看训练 loss，还需要观察验证集 loss、困惑度、生成格式、多样性和样例文本。

### 改动了什么

- 新增验证集评测：
  - `val_loss`
  - `perplexity`
- 新增生成质量指标：
  - `format_accuracy`
  - `distinct_1`
  - `distinct_2`
- 每轮生成样例，保存到：
  - `data/samples/poem_7_epoch_XXX.txt`
- 每轮更新可视化图：
  - `data/figures/training_metrics_7.png`

### 结果

使用本地极小假数据做过完整流程验证：

- 能读取 JSON。
- 能预处理七言诗。
- 能训练 1 轮。
- 能保存 checkpoint。
- 能生成样例。
- 能生成指标图。

## V0.6 Git 与 GitHub 管理

### 为什么这么改

需要能够查看每次代码版本，并把项目同步到 GitHub。

### 改动了什么

- 初始化本地 Git 仓库。
- 分支名设置为 `main`。
- 新增 `.gitignore`，避免提交大文件和缓存：
  - `tangshi/`
  - `data/`
  - `__pycache__/`
  - `*.pt`
  - `*.pth`
  - `*.bin`
- 新增 `sync_to_github.command`。
- 同步脚本绑定 GitHub 仓库：

```text
https://github.com/xiaoyanstudet/LSTM-poetry.git
```

### 结果

由于当前 Codex 沙箱不能直接写 `.git/config`，远程仓库配置需要在正常终端中执行，或通过双击 `sync_to_github.command` 让脚本自动尝试配置。

后续推荐流程：

```bash
git add .
git commit -m "说明本次修改"
git push
```

## V0.7 下载容错增强

### 为什么这么改

用户运行训练时，下载到 `poet.tang.43000.json` 附近出现网络中断：

```text
ConnectionResetError: [Errno 54] Connection reset by peer
```

这是 GitHub raw 网络连接被重置，不是训练逻辑错误。

### 改动了什么

- 下载源从单一 GitHub raw 改为多个镜像：
  - `raw.githubusercontent.com`
  - `cdn.jsdelivr.net`
  - `fastly.jsdelivr.net`
- 新增自动重试：
  - `--download-retries`
- 新增严格下载模式：
  - `--strict-download`
- 默认非严格模式下，如果个别文件暂时下载失败，会使用已经下载成功的 JSON 子集继续训练。
- 下载完成后会检查 JSON 合法性，避免损坏文件进入训练。

### 结果

下次运行时不会重新下载已存在且合法的 JSON 文件，会继续补缺失文件。若网络仍不稳定，默认可以先用已下载数据继续训练。

## V0.8 50 轮训练结果观察

### 为什么记录

用户已经完成 50 轮训练，并观察到生成效果不理想、出现繁体字、格式不稳定和过拟合迹象。

### 实际结果

最新 checkpoint 中的历史指标如下：

```text
train_loss: 6.6854 -> 3.3323
val_loss:   6.2368 -> 5.3418
perplexity: 511.23 -> 208.89
format_accuracy: 0.00%
distinct_1: 52.34%
distinct_2: 95.16%
```

七言训练集规模：

```text
data/poem_7.txt: 10638 首
data/poem_5.txt: 3912 首
```

第 50 轮样例中可以看到：

- 模型已经学到一些唐诗风格词汇和意象。
- 验证集 loss 后期上升，说明已经过拟合。
- 生成格式不稳定，逗号和句号位置经常不符合七言绝句要求。
- 出现繁体字和异体字，如 `煙`、`雲`、`風`、`萬`、`爲`、`歸`。

### 原因分析

繁体字不是模型凭空生成的，而是训练数据本身存在。例如训练集中出现：

```text
煙 670
雲 1513
風 2262
萬 848
爲 1033
歸 1050
閑 699
```

格式不符合的主要原因是生成阶段没有强制七言绝句格式，标点被当作普通字符自由采样。

过拟合的主要原因是：

- 七言数据量只有约 1 万首。
- 当前 LSTM 容量较大，`hidden_dim=600`、`num_layers=2`。
- 训练到 50 轮后，训练 loss 继续下降，但验证 loss 已经回升。

### 结论

继续单纯增加 epoch 不会明显改善质量，反而可能加重过拟合。后续应优先改：

1. 数据预处理阶段加入繁简转换。
2. 生成阶段强制七言格式：
   - 第 8 个字符为 `，`
   - 第 16 个字符为 `。`
   - 第 24 个字符为 `，`
   - 第 32 个字符为 `。`
   - 其他位置禁止生成标点
3. 加入 early stopping，保存验证 loss 最低的模型。
4. 适当降低模型容量或增加正则化：
   - `hidden_dim=384` 或 `512`
   - `weight_decay`
   - 更高 dropout

## V0.9 简体化、格式约束与 Early Stopping

### 为什么这么改

50 轮训练后已经出现过拟合：训练 loss 继续下降，但验证 loss 和 perplexity 回升。同时，生成结果里存在两个明显问题：

- 训练数据来自古籍 JSON，包含大量繁体字和异体字。
- 生成阶段自由采样标点，不能稳定保证 `7字，7字。7字，7字。` 的格式。

因此本版本优先处理数据规范化、生成约束和过拟合控制，而不是继续加大模型或增加 epoch。

### 改动了什么

- 默认 epoch 改为 30：
  - `main.py` 中 `--epochs` 默认值改为 30。
  - `function.TrainConfig.epochs` 默认值改为 30。
- 预处理阶段默认繁转简：
  - 新增 `to_simplified()`。
  - 若安装了 `opencc-python-reimplemented`，优先使用 OpenCC。
  - 若未安装 OpenCC，使用内置常见繁简映射兜底。
- 用户输入首句也会做同样的简体化处理。
- 生成阶段默认强制格式：
  - 五言：第 6、12、18、24 个字符位置强制为 `，。 ，。`。
  - 七言：第 8、16、24、32 个字符位置强制为 `，。 ，。`。
  - 非标点位置禁止采样 `，`、`。`、`、`、`？`、`！` 等标点。
- 增加缓存元信息：
  - 新增 `data/process_meta.json`。
  - 如果预处理版本、简体化设置或数据文件数量变化，会自动重建 `poem_5.txt` 和 `poem_7.txt`。
  - 缓存签名会记录当前简体转换器来源；安装 OpenCC 后，旧的 fallback 简体缓存会自动失效并重建。
- 加入 early stopping：
  - 新增 `--early-stopping-patience`，默认 8。
  - 新增 `--early-stopping-min-delta`，默认 `1e-3`。
  - 验证 loss 连续多轮没有改善时自动停止。
- 保存最佳模型：
  - `data/checkpoints/poem_7_best.pt`
  - `data/checkpoints/poem_7_latest.pt`
- 降低过拟合风险：
  - `hidden_dim` 默认从 600 降为 512。
  - `dropout` 默认从 0.2 提高到 0.3。
  - 新增 `weight_decay=1e-4`。
- 新增参数：
  - `--no-simplify-text`
  - `--no-format-constraint`
  - `--weight-decay`
  - `--early-stopping-patience`
  - `--early-stopping-min-delta`

### 结果

使用本地极小测试数据验证：

```text
原始输入：湖光秋月兩相和，潭面無風鏡未磨。遙望洞庭山水翠，白銀盤裏一青螺。
预处理后：湖 光 秋 月 两 相 和 ， 潭 面 无 风 镜 未 磨 。 遥 望 洞 庭 山 水 翠 ， 白 银 盘 里 一 青 螺 。
format_accuracy=100.00%
best_epoch=1
```

生成样例格式已经被强制稳定为：

```text
湖光秋月两相和，磨面磨磨螺面面。面洞水磨磨螺磨，面洞磨磨面面磨。
```

注意：这是极小假数据验证，只证明代码链路正确。真实效果需要重新处理完整数据并从简体化新数据重新训练。

### 使用方式

默认即可训练 30 轮：

```bash
python main.py
```

如果旧的 `data/poem_7.txt` 是繁体缓存，本版本会自动检测缓存过期并重建。旧的 50 轮 checkpoint 因词表不同会被自动跳过，从新数据重新训练。

## 后续版本计划

### V1.0 生成质量继续优化

计划改动：

- 对重复字、重复短语增加惩罚。
- 增加 beam search 或 nucleus sampling。
- 增加更多人工评分辅助项。
- 对首句续写做更严格的位置对齐。

预期结果：

- 减少机械重复。
- 提高语义连贯性和可读性。

### V1.1 更强模型结构

计划改动：

- 尝试 Attention LSTM。
- 尝试小型 Transformer Decoder。
- 比较 LSTM 与 Transformer 的生成质量。

预期结果：

- 报告中可以形成模型对比实验。
- 判断当前任务是否值得从 LSTM 升级到 Transformer。
