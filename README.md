# 🧠 LLM Tournament

**LLM Tournament** is a research framework to explore the mechanisms that enable cooperative behavior among LLM-based agents. It allows experiments on multi-agent cooperation using a variety of game environments, mechanism designs, and LLM agent architectures.

---

## Project Goals

This project enables:

- Testing LLM agent cooperation in strategic games.
- Comparing the effects of different game types and mechanism structures.
- Evaluating various LLM models under controlled cooperative tasks.

---

## Setup Instructions

Before running the project, follow these steps:

### 1. Use Python 3.10

Make sure you're using Python **3.10**. You can check your version with:

```bash
python3 --version
```

### 2. Virtual Environment

```bash
python3.10 -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
### 4. Install Additional Packages

## Running Experiments
You can use predefined experiment configurations located in the configs/ folder, or define your own in the following YAML format:

```yaml
game:
  type: <GameClassName>
  kwargs:
    <GameParameter>

mechanism:
  type: <MechanismClassName>
  kwargs:
    <MechanismParameter>

agents:
  - llm:
      model: <Model1Name>
      kwargs:
        <Model1Parameter>
    type: <Model1Class>
  - llm:
      model: <Model2Name>
      kwargs:
        <Model2Parameter>
    type: <Model2Class>
  # Add more agents as needed
```

### Run an experiment locally
Use the following command to run a configuration and log the outputs:
```bash
python3 script/run_game.py --config <your_config>.yaml --log
```

### For Vector Server ONLY
To run your experiment as a job on a vector server:
1.	Edit the configuration parameters in run_job.sh and submit.sh.
2.	Submit your job:
```bash
./submit.sh
```

## Folder Structure
```plaintext
llm-tournament/
├── configs/                # Configuration yaml files for experiments
├── script/                 # Scripts for running tournaments
├── src/                    # Source code for the project
│   └── games/              # Game environments and logic used in experiments
│   └── mechanisms/         # Mechanism designs
│   └── agents.py           # LLM agent implementations
├── run_job.sh
└── submit.sh
```
