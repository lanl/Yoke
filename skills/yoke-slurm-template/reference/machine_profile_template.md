# Machine Profile Template

A machine profile records what the `yoke-slurm-template` skill learned about
one cluster, so later runs (by any user) can reuse it. Save it as
`skills/yoke-slurm-template/machines/<ClusterName>.md`, using the
`ClusterName` from `scontrol show config`, in lower case.

Rules:

- Record **machine facts**, which are stable and shared by all users.
- **Allocations** (accounts, QOS access) are per user and change over time.
  Record them only as dated observations under "Allocations seen", and never
  use them without re-querying.
- No credentials, tokens, passwords, or personal home-directory paths. Use
  placeholders such as `$HOME/<venv>`.
- Show the profile (or the diff, when updating) to the user and get approval
  before writing it.
- `tests/skills/test_skill_files.py` checks that each profile has a
  `last verified:` date and the "Allocations seen" section with its warning.

Copy everything below the line into the new file and fill it in.

---

```markdown
# Machine profile: <clustername>

last verified: YYYY-MM-DD
slurm version: <output of `sinfo --version`>
aliases: <other names users call it, login hostnames pattern>

## Machine facts (stable; reuse, spot-check with `sinfo -s`)

### Partitions

| partition | purpose | GPUs/node | max time | default time | notes |
| --- | --- | --- | --- | --- | --- |
| gpu | production | 4x <type> | 16:00:00 | 04:00:00 | |
| gpu_debug | debug | 4x <type> | 02:00:00 | | needs --qos=debug, reservation gpu_debug |

(For a CPU-only machine, write `none` under GPUs/node, and add the quirk line
"CPU-only: harnesses use HarnessTrainer device_type="cpu" and the CPU-only
template".)

### Node hardware (partition: <name>)

- GPUs: <count> x <model>, <memory> GB each, or "none (CPU-only)"
- CPUs: <CPUTot> (<Sockets> sockets x <CoresPerSocket> cores x
  <ThreadsPerCore> threads)
- Memory: <RealMemory> MB

### Recommended job settings

- `--cpus-per-task`: <n> (even values only, if ThreadsPerCore is 2)
- `OMP_NUM_THREADS`: <n>
- `--mem-per-gpu`: <value or "not needed">
- binding flags: <e.g. --gpu-bind=closest, or "none">
- `NCCL_SOCKET_IFNAME`: <e.g. hsn0> (checked on a compute node: yes/no), or
  "not needed (CPU-only)"
- QOS needed for production: <name or "none">

### Debug pattern

<partition / qos / reservation lines that work for short test jobs>

### Known quirks

- <e.g. "login nodes have no GPUs; ip link on login differs from compute">

## Environment hints (semi-stable)

- Module loads: <e.g. `module load cudatoolkit/12.4` or "none">
- Activation line shape: `source $HOME/<venv>/bin/activate` or
  `conda activate <env>`
- Notes: <e.g. "CUDA 12.6 wheels needed for torch">

## Allocations seen (volatile: re-query with sacctmgr before use)

As of YYYY-MM-DD, for one user. Accounts and QOS access differ between users
and change over time. Always re-run
`sacctmgr -nP show assoc user=$USER format=cluster,account,partition,qos`.

| observed | account | partitions | qos |
| --- | --- | --- | --- |
| YYYY-MM-DD | <account> | <all or list> | <qos list> |

## Templates generated with this profile

- <path to training_slurm.tmpl> (YYYY-MM-DD)
```
