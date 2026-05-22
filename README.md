# train-logger

[![PyPI version](https://img.shields.io/pypi/v/train-logger)](https://pypi.org/project/train-logger/)
[![Python](https://img.shields.io/pypi/pyversions/train-logger)](https://pypi.org/project/train-logger/)
[![CI](https://github.com/byeolki/discord-train-logger/actions/workflows/ci.yml/badge.svg)](https://github.com/byeolki/discord-train-logger/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Multi-backend training notifications for ML jobs.
Connect **Discord**, **Slack**, **Telegram**, and **Console** — all from one logger.

## Installation

```bash
pip install train-logger
```

## Quick Start

```python
from train_logger import TrainLogger, DiscordBackend, SlackBackend, ConsoleBackend

logger = TrainLogger(
    DiscordBackend(webhook_url="https://discord.com/api/webhooks/..."),
    SlackBackend(webhook_url="https://hooks.slack.com/services/..."),
    ConsoleBackend(),  # always available — no webhook needed
)

logger.on_train_start("my-experiment", config={"epochs": 100, "lr": 3e-4})

for epoch in range(1, 101):
    # ... train ...
    logger.on_epoch_end(epoch, {"loss": 0.42, "val_loss": 0.38}, total_epochs=100, step=epoch * 500)

    if new_best:
        logger.on_best_metric("val_loss", best_val_loss, step=global_step)

logger.on_train_end(total_steps=50000, best_val_loss=0.31)
```

## Backends

### 알림 채널

| Backend | Install | Notes |
|---|---|---|
| `DiscordBackend(webhook_url)` | built-in | Server Settings → Integrations → Webhooks |
| `SlackBackend(webhook_url)` | built-in | Slack Incoming Webhooks |
| `TelegramBackend(token, chat_id)` | built-in | Bot via @BotFather |
| `ConsoleBackend()` | built-in | Prints to stdout. No setup needed |

### 실험 추적 플랫폼

| Backend | Install | Notes |
|---|---|---|
| `WandbBackend(project)` | `pip install train-logger[wandb]` | metrics, config, summary, alerts |
| `MLflowBackend(experiment_name)` | `pip install train-logger[mlflow]` | metrics, params, tags, artifacts |
| `CometBackend(project_name)` | `pip install train-logger[comet]` | metrics, parameters, models |
| `NeptuneBackend(project)` | `pip install train-logger[neptune]` | metrics, metadata, artifacts |
| `TensorBoardBackend(log_dir)` | `pip install train-logger[tensorboard]` | scalars, text (SummaryWriter) |
| `AimBackend(experiment)` | `pip install train-logger[aim]` | metrics, metadata (local repo) |

모두 한 번에 설치:
```bash
pip install train-logger[all]
```

All backends can be combined. Events are fanned out to all of them simultaneously.

```python
# Add backends dynamically
logger = TrainLogger()
logger.add_backend(DiscordBackend(...))
logger.add_backend(TelegramBackend(token="...", chat_id="..."))
```

## 실험 추적 플랫폼

### Weights & Biases

```bash
pip install train-logger[wandb]
```

```python
from train_logger import TrainLogger, WandbBackend, DiscordBackend

logger = TrainLogger(
    WandbBackend(project="my-project"),   # auto-calls wandb.init()
    DiscordBackend(webhook_url="..."),    # fan-out to Discord simultaneously
)

logger.on_train_start("my-experiment", config={"lr": 3e-4, "epochs": 100})

for epoch in range(1, 101):
    logger.on_epoch_end(epoch, {"loss": 0.42, "val_loss": 0.38}, total_epochs=100, step=epoch * 500)

logger.on_train_end(total_steps=50000, best_val_loss=0.31)
```

Reuse an already-active run:

```python
import wandb
wandb.init(project="my-project", name="run-1")

logger = TrainLogger(WandbBackend(run=wandb.run))
```

What gets logged to wandb:

| Event | wandb action |
|---|---|
| `on_train_start` | `wandb.init()` with config (or updates existing run config) |
| `on_epoch_end` / `on_step_end` | `wandb.log(metrics, step=step)` |
| `on_best_metric` | `wandb.log({"best/<name>": value})` + `run.summary` update |
| `on_checkpoint_saved` | `run.summary` — last checkpoint path and step |
| `on_train_end` | `run.summary` — total steps and best val loss, then `wandb.finish()` |
| `on_error` | `wandb.alert(level=ERROR)` |
| `send` / `summarize` | `wandb.alert(level=INFO)` |

## API Reference

### `TrainLogger(*backends)`

| Method | Description |
|--------|-------------|
| `on_train_start(experiment, config)` | Green — training started with config fields |
| `on_epoch_end(epoch, metrics, total_epochs, step)` | Blue — metrics + Unicode progress bar |
| `on_step_end(step, metrics, total_steps)` | Blue — step-level metrics with progress bar |
| `on_best_metric(metric_name, value, step)` | Yellow — new best metric alert |
| `on_checkpoint_saved(step, path)` | Gray — checkpoint saved |
| `on_train_end(total_steps, best_val_loss)` | Green — training complete summary |
| `on_error(error, context)` | Red — exception with context |
| `send(message, color)` | Custom one-line message |
| `send_file(file_path, message, filename)` | File attachment (audio, images, etc.) |
| `catch_errors(context)` | Context manager — auto-posts `on_error` on exception |

Available colors: `green`, `red`, `yellow`, `blue`, `gray`, `orange`

### `catch_errors` context manager

```python
with logger.catch_errors("train_step"):
    loss = model.train_step(batch)
# If an exception is raised: posts on_error, then re-raises
```

### MLflow

```bash
pip install train-logger[mlflow]
```

```python
from train_logger import TrainLogger, MLflowBackend

logger = TrainLogger(
    MLflowBackend(experiment_name="my-project", tracking_uri="http://localhost:5000"),
)
# 또는 기본 로컬 저장 (./mlruns)
logger = TrainLogger(MLflowBackend())
```

`on_train_start` 때 자동으로 run을 시작하고 config를 param으로 기록합니다.
`on_train_end` 때 `mlflow.end_run()`을 호출합니다.

### Comet ML

```bash
pip install train-logger[comet]
```

```python
from train_logger import TrainLogger, CometBackend

logger = TrainLogger(
    CometBackend(api_key="...", project_name="my-project", workspace="my-workspace"),
)
# 기존 experiment 재사용
import comet_ml
exp = comet_ml.Experiment(api_key="...", project_name="...")
logger = TrainLogger(CometBackend(experiment=exp))
```

### Neptune

```bash
pip install train-logger[neptune]
```

```python
from train_logger import TrainLogger, NeptuneBackend

logger = TrainLogger(
    NeptuneBackend(project="workspace/project", api_token="..."),
)
# 기존 run 재사용
import neptune
run = neptune.init_run(project="workspace/project")
logger = TrainLogger(NeptuneBackend(run=run))
```

metrics는 `run["metrics/<name>"].append(value, step=step)` 형태로 기록됩니다.

### TensorBoard

```bash
pip install train-logger[tensorboard]
# 또는 PyTorch와 함께 사용 시 추가 설치 불필요 (torch.utils.tensorboard 사용)
```

```python
from train_logger import TrainLogger, TensorBoardBackend

logger = TrainLogger(
    TensorBoardBackend(log_dir="runs/my-experiment"),
)
# 기존 writer 재사용
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs/my-experiment")
logger = TrainLogger(TensorBoardBackend(writer=writer))
```

### Aim

```bash
pip install train-logger[aim]
```

```python
from train_logger import TrainLogger, AimBackend

logger = TrainLogger(
    AimBackend(experiment="my-experiment", repo="."),
)
# 기존 run 재사용
from aim import Run
run = Run(experiment="my-experiment")
logger = TrainLogger(AimBackend(run=run))
```

---

## AI Summary (OpenAI)

Pass an `AISummarizer` to get a 1–2 sentence natural language analysis of your training run,
posted directly to all your backends.

```python
from train_logger import TrainLogger, DiscordBackend, AISummarizer

logger = TrainLogger(
    DiscordBackend(webhook_url="..."),
    summarizer=AISummarizer(api_key="sk-...", model="gpt-4o-mini"),
)

# Events are recorded automatically as you log
logger.on_train_start("my-exp", config={"lr": 3e-4, "epochs": 100})
for epoch in range(1, 101):
    logger.on_epoch_end(epoch, {"loss": 0.42, "val_loss": 0.38}, total_epochs=100)

# Posts an orange embed like:
# "Training is converging steadily — val_loss dropped from 0.89 to 0.38
#  over 100 epochs with no signs of overfitting."
logger.summarize()
```

Call `summarize()` at any point (mid-training, on checkpoint, at the end).
No extra packages required — uses `requests` directly against the OpenAI API.

## HuggingFace Transformers Callback

No extra dependencies — just pass the callback to `Trainer`.

```python
from transformers import Trainer
from train_logger import TrainLogger, DiscordBackend, HuggingFaceCallback

logger = TrainLogger(DiscordBackend(webhook_url="..."))
trainer = Trainer(
    model=model,
    args=training_args,
    callbacks=[HuggingFaceCallback(logger, experiment="my-run")],
)
trainer.train()
```

Hooks: `on_train_begin` → `on_log` (each loss log) → `on_save` → `on_train_end`

## PyTorch Lightning Callback

```python
import pytorch_lightning as pl
from train_logger import TrainLogger, DiscordBackend, LightningCallback

logger = TrainLogger(DiscordBackend(webhook_url="..."))
trainer = pl.Trainer(
    max_epochs=100,
    callbacks=[LightningCallback(logger, experiment="my-run")],
)
trainer.fit(model)
```

Hooks: `on_train_start` → `on_train_epoch_end` → `on_save_checkpoint` → `on_train_end` → `on_exception`

## PyPI Release

Tag a commit to trigger automated publish:

```bash
git tag v0.1.0
git push origin v0.1.0
```

Requires a PyPI [Trusted Publisher](https://docs.pypi.org/trusted-publishers/) under the `release` GitHub environment.

## License

MIT
