# HELP Guide

This document provides a quick-start guide to manage Conda environments, export your dependencies, and run long training sessions remotely via SSH using `tmux`.

---

## Conda Environment Basics

### 1. Create a new Conda Environment

```bash
conda create -n your_env_name python=3.10
```

### 2. Activate your Environment

```bash
conda activate your_env_name
```

### 3. Install Packages

```bash
conda install package_name
# or use pip
pip install package_name
```

### 4. Deactivate your Environment

```bash
conda deactivate
```

---

## Export Your Environment to a requirements.txt File

Once your environment has all necessary packages installed, you can export them:

```bash
pip freeze > requirements.txt
```

To install these requirements again (e.g., on another machine):

```bash
pip install -r requirements.txt
```

---

## Using tmux for Remote Python Sessions

### 1. Install tmux

- Ubuntu/Debian:
```bash
sudo apt-get install tmux
```
- CentOS/Fedora:
```bash
sudo yum install tmux
```

### 2. Basic tmux Usage

Start a new session:
```bash
tmux new -s my_training_session
```

Detach from the session (while your training runs):
- Press `Ctrl+b`, then `d`.

List existing tmux sessions:
```bash
tmux ls
```

Reattach to your session:
```bash
tmux attach-session -t my_training_session
```

### 3. Run your Python Training Script

Inside a `tmux` session:

```bash
conda activate your_env_name
python utils/trainingMyCrazyModel.py
```

Now you can safely detach from your session and let your training continue uninterrupted.

---

