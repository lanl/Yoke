# Read-Only SLURM Queries

Commands for finding out what a cluster offers and what the current user
may use. All of them are **read-only**. Show each command to the user before
running it. **Never** run `sbatch`, `salloc`, `srun`, `scancel`, `scontrol
update`, or any `sacctmgr` command other than `show`/`list`.

Output formats differ between sites and SLURM versions. If a command fails or
prints nothing useful, say so and ask the user. Don't guess.

## 0. Availability and version

```bash
command -v sinfo scontrol sacctmgr squeue
sinfo --version
```

If `sinfo` is missing, you are not on a SLURM login node. See the fallbacks in
`SKILL.md`.

## 1. Cluster identity

```bash
scontrol show config | grep -iE "^(ClusterName|SlurmctldHost|SelectType|GresTypes)"
hostname
```

`ClusterName` names the machine profile file:
`skills/yoke-slurm-template/machines/<ClusterName>.md`.

## 2. Partitions

```bash
sinfo -s                                   # partition, avail, timelimit, nodes A/I/O/T
sinfo -o "%P %a %l %D %c %m %G %f"         # + CPUs, memory (MB), GRES, features
scontrol show partition <name>             # MaxTime, DefaultTime, QoS, AllowAccounts
```

What to read off:

- which partitions have GPUs (`%G` shows something like `gpu:4` or
  `gpu:a100:4`);
- each partition's `MaxTime` (the wall-time ceiling) and `DefaultTime`;
- `AllowAccounts`/`AllowQos`/`QoS` restrictions;
- debug or test partitions (often `*_debug`), which have short limits and
  quick starts.

## 3. Node hardware

Pick one node from the chosen partition (`sinfo -p <partition> -N -h -o %N |
head -1`), then:

```bash
scontrol show node <node>
```

Read: `CPUTot`, `Sockets`, `CoresPerSocket`, `ThreadsPerCore`, `RealMemory`
(MB), `Gres` (GPU type and count), `ActiveFeatures`.

Derived defaults for the interview:

- GPUs per node = the count in `Gres`. A `Gres` of `(null)` (and `%G` of
  `(null)` in `sinfo`) means a **CPU-only** partition: use the CPU-only
  template, skip the GPU, memory-per-GPU and NCCL items below, and base the
  thread defaults on cores per node.
- `--cpus-per-task` = `CPUTot / GPUs per node`, rounded down to an even
  number when `ThreadsPerCore == 2` (existing templates note that odd values
  are rounded up by SLURM).
- `OMP_NUM_THREADS` = physical cores per task (`--cpus-per-task /
  ThreadsPerCore`), or the value in the saved profile.
- `--mem-per-gpu` = about `RealMemory / GPUs per node`, minus a margin, only
  if the site requires a memory request.

## 4. The user's allocations (always re-query)

```bash
sacctmgr -nP show assoc user=$USER format=cluster,account,partition,qos
sacctmgr -nP show qos format=name,maxwall,maxtrespu,maxjobspu,priority
```

- `account` values are what goes in `#SBATCH --account=` (or the CSV
  `allocation` column).
- A blank `partition` in an association means all partitions.
- QOS limits (`maxwall`) can be stricter than the partition's `MaxTime`.

Allocations differ between users and change over time. Do **not** take them
from a saved profile. Query them every time.

If `sacctmgr` is restricted at the site, ask the user for their account
names, or suggest a site tool they may know (many sites have a
`myaccounts`-style command).

## 5. Reservations (debug variants)

```bash
scontrol show reservation
```

Note GPU debug reservations (for example `gpu_debug`) for the commented-out
debug block. Check the debug QOS's flags too
(`sacctmgr -nP show qos <name> format=name,flags`). With `RequiresReservation`,
debug jobs must pass `--reservation=<name>`, or sbatch fails with
`Invalid qos specification`. A reservation's `Accounts=` list also admits
child accounts, so validate with `sbatch --test-only <script>` (read-only:
it schedules nothing).

## 6. Network interface for NCCL

(GPU templates only. CPU mode uses `gloo` on the loopback interface and needs
no NCCL setting.)

`NCCL_SOCKET_IFNAME` must name a compute-node interface (for example `hsn0`
on HPE Slingshot, `ib0` on InfiniBand). The login node may differ, so:

- use the saved profile's value if there is one;
- otherwise ask the user, or suggest checking `ip -br link` **on a compute
  node** in their next interactive session. Don't start one yourself.

## 7. Environment

Ask how the user activates their Yoke environment on this machine (module
loads, `source <venv>/bin/activate`, `conda activate <env>`). Read-only
helpers, if the user agrees:

```bash
module avail 2>&1 | head -50     # only if `module` exists
echo $CONDA_PREFIX $VIRTUAL_ENV
```

## Summary to present

After querying, show a short table: partitions (GPU, time limit, nodes), node
hardware, the user's accounts and QOS with limits, debug options, and the
suggested defaults. Then interview the user.
