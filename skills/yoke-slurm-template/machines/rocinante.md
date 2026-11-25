# Machine profile: rocinante

last verified: 2026-11-13
slurm version: 25.11.8
aliases: roci; compute nodes ro[1000-1511]

## Machine facts (stable; reuse, spot-check with `sinfo -s`)

### Partitions

| partition | purpose | GPUs/node | max time | default time | notes |
| --- | --- | --- | --- | --- | --- |
| standard | production (default) | none | UNLIMITED (QOS caps it) | 01:00:00 | 376 DDR nodes; whole-node allocation; denies QOS `debug` |
| debug | short tests | none | UNLIMITED (QOS caps it) | 01:00:00 | same nodes as standard; allows QOS `debug`, `standby`; higher priority tier |
| hbm | HBM-memory nodes | none | UNLIMITED | | 128 nodes, about 110 GB HBM |
| hbm_debug | HBM debug | none | UNLIMITED | | |
| ci | CI | none | UNLIMITED | | same nodes as standard |
| shasta_dev | system development | none | | | not for users |

All partitions are CPU-only (`sinfo %G` is `(null)`), and `standard`/`debug`
use `OverSubscribe=EXCLUSIVE`: a job gets whole nodes.

### Node hardware (partition: standard)

- GPUs: none (CPU-only)
- CPUs: 224 (2 sockets x 56 cores x 2 threads), Intel Xeon Platinum 8480+
  (Sapphire Rapids)
- Memory: 237000 MB

### Recommended job settings

- `--nodes=1 --ntasks=1` (Yoke CPU mode is one process per job)
- `--cpus-per-task`: 224 (the node is exclusive anyway)
- `OMP_NUM_THREADS`: small (about 4) for small models; up to 112 (physical
  cores) for compute-heavy ones. Time it in a smoke run.
- `--mem-per-gpu`: not needed
- binding flags: none
- `NCCL_SOCKET_IFNAME`: not needed (CPU-only)
- QOS needed for production: `standard` (`--qos=standard`)

### Debug pattern

```
#SBATCH --partition=debug
#SBATCH --qos=debug
#SBATCH --time=00:30:00
```

```
#SBATCH --reservation=debug
```

The `debug` QOS has the `RequiresReservation` flag. Without
`--reservation=debug`, sbatch fails with `Invalid qos specification`. The
`debug` reservation lists accounts `systems,asc`, and child accounts of `asc`
(e.g. `xd`) may use it. This was checked with `sbatch --test-only` on
2026-11-13. QOS `debug`: 2 h maxwall, 4 nodes, 1 running job per user (as of
2026-11-13). A template shared by debug and production rows can take the
whole option from a CSV column: `#SBATCH <RESERVATION_OPT>`, with
`--reservation=debug` for debug rows and `--reservation=` (none) for
`standard`.

### Known quirks

- CPU-only: harnesses use `HarnessTrainer(..., device_type="cpu")` and the
  CPU-only template (`skills/yoke-slurm-template/reference/slurm_template_skeleton.md`).
- `standard` and `debug` allocate whole nodes, so each Yoke study row holds
  one full node for its job.
- QOS `standard`: 12 h maxwall, 50 nodes per user (as of 2026-11-13).
- `debug` QOS needs `--reservation=debug` (see Debug pattern).

## Environment hints (semi-stable)

- Module loads: `module load python/3.13.18` (the base interpreter of a
  `python -m venv` environment)
- Activation line shape: `source <repo>/.venv/bin/activate`
- Notes: CPU torch wheels work; no CUDA modules are needed.

## Allocations seen (volatile: re-query with sacctmgr before use)

As of 2026-11-13, for one user. Accounts and QOS access differ between users
and change over time. Always re-run
`sacctmgr -nP show assoc user=$USER format=cluster,account,partition,qos`.

| observed | account | partitions | qos |
| --- | --- | --- | --- |
| 2026-11-13 | xd | standard; debug; hbm_debug | standard,standby; debug,standby; hbm_debug,standby |

## Templates generated with this profile

- applications/harnesses/ro_tpbf_mlp/training_slurm.tmpl (2026-11-13)
