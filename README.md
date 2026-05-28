# ASHi

[![PyPI version](https://img.shields.io/pypi/v/ashi)](https://pypi.org/project/ashi/)
[![Python](https://img.shields.io/pypi/pyversions/ashi)](https://pypi.org/project/ashi/)
[![CI](https://github.com/byeolki/ashi/actions/workflows/ci.yml/badge.svg)](https://github.com/byeolki/ashi/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**One logger. Every destination.**

ASHi fans out ML training events to every platform you care about — notification channels and experiment trackers alike — from a single, uniform API. Stop wiring up each platform separately; call `on_epoch_end()` once and every backend receives it simultaneously.

## Features

- **Notification channels** — Discord, Slack, Telegram, Console
- **Experiment tracking** — Weights & Biases, MLflow, Comet ML, Neptune, TensorBoard, Aim
- **Framework callbacks** — HuggingFace Transformers, PyTorch Lightning (zero extra dependencies)
- **AI summaries** — GPT-powered one-sentence training summaries posted directly to all backends
- **Fan-out by default** — every event goes to every backend in one call
- **Resilient** — a failing backend never crashes training; each returns `bool` success
- **Composable** — mix and match any backends; add more at runtime with `add_backend()`

## Installation

```bash
pip install ashi
```

Install optional experiment-tracking backends:

```bash
pip install ashi[wandb]       # Weights & Biases
pip install ashi[mlflow]      # MLflow
pip install ashi[comet]       # Comet ML
pip install ashi[neptune]     # Neptune.ai
pip install ashi[tensorboard] # TensorBoard
pip install ashi[aim]         # Aim

pip install ashi[all]         # everything at once
```

## Quick Start

```python
from ashi import TrainLogger, DiscordBackend, WandbBackend, ConsoleBackend

logger = TrainLogger(
    DiscordBackend(webhook_url="https://discord.com/api/webhooks/..."),
    WandbBackend(project="my-project"),
    ConsoleBackend(),
)

logger.on_train_start("my-experiment", config={"epochs": 100, "lr": 3e-4})

for epoch in range(1, 101):
    # ... train ...
    logger.on_epoch_end(epoch, {"loss": 0.42, "val_loss": 0.38}, total_epochs=100, step=epoch * 500)

    if new_best:
        logger.on_best_metric("val_loss", best_val_loss, step=global_step)

logger.on_train_end(total_steps=50000, best_val_loss=0.31)
```

Every call fans out to Discord, W&B, and the console simultaneously.

## Backends

### Notification Channels

| Backend | Setup |
|---|---|
| `DiscordBackend(webhook_url)` | Server Settings → Integrations → Webhooks |
| `SlackBackend(webhook_url)` | [Slack Incoming Webhooks](https://api.slack.com/messaging/webhooks) |
| `TelegramBackend(token, chat_id)` | Create a bot via @BotFather, get `chat_id` from @userinfobot |
| `ConsoleBackend()` | No setup — prints to stdout with ANSI colors |

### Experiment Tracking Platforms

| Backend | Install | What gets logged |
|---|---|---|
| `WandbBackend(project)` | `ashi[wandb]` | metrics, config, run summary, alerts |
| `MLflowBackend(experiment_name)` | `ashi[mlflow]` | metrics, params, tags |
| `CometBackend(project_name)` | `ashi[comet]` | metrics, parameters |
| `NeptuneBackend(project)` | `ashi[neptune]` | metrics, metadata |
| `TensorBoardBackend(log_dir)` | `ashi[tensorboard]` | scalars, text |
| `AimBackend(experiment)` | `ashi[aim]` | metrics, metadata |

Mix backends freely:

```python
logger = TrainLogger(
    DiscordBackend(webhook_url="..."),
    WandbBackend(project="..."),
    MLflowBackend(experiment_name="..."),
    ConsoleBackend(),
)
```

Or add them after construction:

```python
logger = TrainLogger()
logger.add_backend(DiscordBackend(webhook_url="..."))
logger.add_backend(WandbBackend(project="..."))
```

## API Reference

### `TrainLogger(*backends, summarizer=None)`

| Method | Description |
|--------|-------------|
| `on_train_start(experiment, config)` | Training started — logs config to trackers, posts to notification channels |
| `on_epoch_end(epoch, metrics, total_epochs, step)` | Metrics + Unicode progress bar |
| `on_step_end(step, metrics, total_steps)` | Step-level metrics + progress bar |
| `on_best_metric(metric_name, value, step)` | New best metric alert |
| `on_checkpoint_saved(step, path)` | Checkpoint saved notification |
| `on_train_end(total_steps, best_val_loss)` | Training complete summary |
| `on_error(error, context)` | Exception with context — sends alert to all backends |
| `send(message, color)` | Custom one-line message |
| `send_file(file_path, message, filename)` | File attachment (images, plots, model weights, etc.) |
| `catch_errors(context)` | Context manager — auto-calls `on_error` on exception, then re-raises |
| `summarize()` | Call the AI summarizer and post the result to all backends |

Available colors for `send()`: `green`, `red`, `yellow`, `blue`, `gray`, `orange`

### `catch_errors` context manager

```python
with logger.catch_errors("eval_loop"):
    metrics = model.evaluate(val_dataloader)
# Any exception → on_error posted to all backends, exception re-raised
```

---

## Experiment Tracking Details

### Weights & Biases

```python
from ashi import TrainLogger, WandbBackend

logger = TrainLogger(WandbBackend(project="my-project"))
# wandb.init() is called automatically on on_train_start
```

Reuse an existing run:

```python
import wandb
wandb.init(project="my-project", name="run-1")
logger = TrainLogger(WandbBackend(run=wandb.run))
```

| Event | wandb action |
|---|---|
| `on_train_start` | `wandb.init()` with config, or updates existing run config |
| `on_epoch_end` / `on_step_end` | `wandb.log(metrics, step=step)` |
| `on_best_metric` | `wandb.log({"best/<name>": value})` + `run.summary` update |
| `on_checkpoint_saved` | `run.summary` — last checkpoint path and step |
| `on_train_end` | `run.summary` update, then `wandb.finish()` |
| `on_error` | `wandb.alert(level=ERROR)` |
| `send` / `summarize` | `wandb.alert(level=INFO)` |

Pass `finish_on_end=False` to keep the run open after `on_train_end`.

### MLflow

```bash
pip install ashi[mlflow]
```

```python
from ashi import TrainLogger, MLflowBackend

# Local ./mlruns directory
logger = TrainLogger(MLflowBackend())

# Remote tracking server
logger = TrainLogger(
    MLflowBackend(experiment_name="my-project", tracking_uri="http://localhost:5000")
)
```

`on_train_start` starts a new run and logs config as params. `on_train_end` calls `mlflow.end_run()`.

### Comet ML

```bash
pip install ashi[comet]
```

```python
from ashi import TrainLogger, CometBackend

logger = TrainLogger(
    CometBackend(api_key="...", project_name="my-project", workspace="my-workspace")
)

# Reuse an existing experiment
import comet_ml
exp = comet_ml.Experiment(api_key="...", project_name="...")
logger = TrainLogger(CometBackend(experiment=exp))
```

### Neptune

```bash
pip install ashi[neptune]
```

```python
from ashi import TrainLogger, NeptuneBackend

logger = TrainLogger(
    NeptuneBackend(project="workspace/project", api_token="...")
)

# Reuse an existing run
import neptune
run = neptune.init_run(project="workspace/project")
logger = TrainLogger(NeptuneBackend(run=run))
```

Metrics are logged via `run["metrics/<name>"].append(value, step=step)`.

### TensorBoard

```bash
pip install ashi[tensorboard]
```

```python
from ashi import TrainLogger, TensorBoardBackend

logger = TrainLogger(TensorBoardBackend(log_dir="runs/my-experiment"))

# Reuse an existing SummaryWriter
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs/my-experiment")
logger = TrainLogger(TensorBoardBackend(writer=writer))
```

Works with `torch.utils.tensorboard` (no extra install if PyTorch is already present) or `tensorboardX`.

### Aim

```bash
pip install ashi[aim]
```

```python
from ashi import TrainLogger, AimBackend

logger = TrainLogger(AimBackend(experiment="my-experiment", repo="."))

# Reuse an existing run
from aim import Run
run = Run(experiment="my-experiment")
logger = TrainLogger(AimBackend(run=run))
```

---

## Framework Callbacks

### HuggingFace Transformers

No extra dependencies — pass the callback to `Trainer`.

```python
from transformers import Trainer
from ashi import TrainLogger, DiscordBackend, HuggingFaceCallback

logger = TrainLogger(DiscordBackend(webhook_url="..."))
trainer = Trainer(
    model=model,
    args=training_args,
    callbacks=[HuggingFaceCallback(logger, experiment="my-run")],
)
trainer.train()
```

Hooks: `on_train_begin` → `on_log` (each loss log) → `on_save` → `on_train_end`

### PyTorch Lightning

```python
import pytorch_lightning as pl
from ashi import TrainLogger, DiscordBackend, LightningCallback

logger = TrainLogger(DiscordBackend(webhook_url="..."))
trainer = pl.Trainer(
    max_epochs=100,
    callbacks=[LightningCallback(logger, experiment="my-run")],
)
trainer.fit(model)
```

Hooks: `on_train_start` → `on_train_epoch_end` → `on_save_checkpoint` → `on_train_end` → `on_exception`

---

## AI Summary (OpenAI)

Pass an `AISummarizer` to get a 1–2 sentence natural-language analysis of your training run, posted to all backends automatically.

```python
from ashi import TrainLogger, DiscordBackend, AISummarizer

logger = TrainLogger(
    DiscordBackend(webhook_url="..."),
    summarizer=AISummarizer(api_key="sk-...", model="gpt-4o-mini"),
)

logger.on_train_start("my-exp", config={"lr": 3e-4, "epochs": 100})
for epoch in range(1, 101):
    logger.on_epoch_end(epoch, {"loss": 0.42, "val_loss": 0.38}, total_epochs=100)

# Posts an orange message like:
# "Training is converging steadily — val_loss dropped from 0.89 to 0.38
#  over 100 epochs with no signs of overfitting."
logger.summarize()
```

Call `summarize()` at any point — mid-training, on checkpoint, or at the end. No extra packages required beyond `requests`.

---

## License

MIT
