# Study CSV and Template Rules

How `yoke-start-study` (`yoke.cli.start_study`) and
`yoke.harnesses.base.HarnessStudy` turn a harness directory into
`runs/study_###/` directories, and how continuation re-renders them. Knowing
these rules prevents silent mistakes: a misspelled `<KEY>` is **not** an
error, it just stays in the file.

## The study CSV

Read with
`pd.read_csv(csv, sep=",", header=0, index_col=0, comment="#", engine="python")`.

- The **first column** is the study index. Name it `studyIDX`. Each row
  becomes `runs/study_<NNN>/`.
- Every other column is a substitution key. Column `FOO` replaces `<FOO>` in
  the templates.
- Lines starting with `#` are comments. A `#` inside a value cuts the value
  off.
- Don't put spaces after commas: string values would keep them.
- Empty cells become NaN and render as `nan`. Fill every cell.
- An all-integer column renders as integers. A float column renders with
  Python's `str` (for example `0.0001`, `1e-05`).
- Conventional columns: `KNODES`, `NGPUS` (used by GPU SLURM templates; a
  CPU-mode template runs one task on one node and needs neither),
  `train_script` (the script filename, also listed in `cp_files.txt`), and
  `allocation` (the SLURM account, if the template uses `<allocation>`).
- Keep a small **smoke-test CSV** (2–3 rows, few batches) next to the
  production one, as `ch_ldrViT_lsc_cyl/ldrViT_2frame_lsc_cyl_smoke.csv`
  does, with `#` comment lines explaining each row.

## Value formatting (`yoke.helpers.strings.replace_keys`)

| Key or type | Rendered as |
| --- | --- |
| `studyIDX` | 3 digits, zero-padded (`007`) |
| float | `str(value)` |
| bool | `True` / `False` |
| int | `%d` |
| str | as is |
| anything else | `ValueError` |

## `training_input.tmpl`

The train script's `@`-file: one argparse token per line (flag on one line,
value on the next; a list flag is followed by one value per line).

```
--studyIDX
<studyIDX>
--FILELIST_DIR
/path/on/the/cluster/applications/filelists/
--some_flag
<SOME_KEY>
--trn_rcrd_filename
./training_study<studyIDX>_epoch<epochIDX>.csv
--val_rcrd_filename
./validation_study<studyIDX>_epoch<epochIDX>.csv
--total_epochs
100
--cycle_epochs
3
# <<optional:CONTINUATION>>
--continuation
--checkpoint
<CHECKPOINT>
# <<end>>
```

- **Required**: the `# <<optional:CONTINUATION>>` block at the end, exactly as
  above.
- Optional blocks: lines between `# <<optional:KEY>>` and `# <<end>>` are kept
  only if `KEY` is present in the substitutions (presence, not truth). The
  marker lines are never written out. Blocks can't be nested. Only input
  and evaluation templates support them; the training submission template
  does not.
- Machine-specific paths (data and filelist directories) live here, never in
  `src/yoke`.
- Values that never vary are written literally. Values that vary per study
  are `<KEY>`s with a CSV column.
- Every flag must exist in the train script's parser. The `@`-file is parsed
  by argparse, so a typo is an error at job start.

## Submission template

`training_slurm.tmpl` (made by `yoke-slurm-template`) or `training_shell.tmpl`.
Plain `<KEY>` substitution, no optional blocks. Required tokens:
`<studyIDX>`, `<epochIDX>`, `<train_script>`, `<INPUTFILE>`, plus `<KNODES>`
and `<NGPUS>` for a GPU SLURM template. A CPU-mode SLURM template
(`device_type="cpu"`) needs only the first four (and `<allocation>`, if
used).

For `shell` submission, the script is run with `source`. Its job is to run
`python <train_script> @<INPUTFILE>` (see
`applications/harnesses/moving_mnist/training_shell.tmpl`). Continuation
resubmits with the type in `args.submissionType`, which defaults to `slurm`.
The flag comes from `yoke.helpers.cli.add_default_args`, so a shell harness
must add `--submissionType` / `shell` to `training_input.tmpl`.

## What happens at study creation (`HarnessStudy.run_study`)

For each CSV row, in `runs/study_NNN/`:

1. Copy every line of `cp_files.txt` (paths relative to the harness dir)
   into the study dir. The train script and any evaluator script must be
   listed.
2. Write the **continuation templates** `training_input.tmpl` and the
   submission template, with CSV keys substituted. `<epochIDX>`,
   `<INPUTFILE>`, and `<CHECKPOINT>` are left as is, and the CONTINUATION
   block is kept.
3. If evaluation is configured, write `evaluation_input.tmpl` and the
   evaluation submission template, with `<CHECKPOINT>`, `<INPUTFILE>`,
   `<STEM>`, and `<epochIDX>` left as is.
4. Write the first-launch files `studyNNN_START.input` and
   `studyNNN_START.<slurm|sh>`, with `epochIDX=0001` and
   `INPUTFILE=studyNNN_START.input`, and the CONTINUATION block dropped.
5. Submit (`cd study_NNN; sbatch|source studyNNN_START.<ext>`), or with
   `--dryrun` print `[DRY RUN] Would execute: ...` instead. Files are
   written either way.

## Continuation (`HarnessStudy.continuation_setup`)

At the end of each cycle, the trainer (rank 0, inside the study dir):

- writes `studyNNN_restart_training_epochMMMM.input` from the study's
  `training_input.tmpl`, replacing only `<CHECKPOINT>`;
- writes `studyNNN_restart_training_epochMMMM.<ext>` from the study's
  submission template, replacing `<INPUTFILE>` and `<epochIDX>`;
- submits it.

So every other `<KEY>` must already be filled in at study creation. The
literal `<epochIDX>` left in a restart `.input` file is fine for record
filenames, because the epoch functions replace it at run time.

## Evaluation (optional)

Enabled when the harness has both `evaluation_input.tmpl` and
`evaluation_<slurm|shell>.tmpl`. Only one of them is a `ValueError`. Also
pass `evaluate_after_training=True` to `HarnessTrainer`.

- Tokens: CSV keys, `<CHECKPOINT>`, `<INPUTFILE>`, `<STEM>` (the checkpoint
  file stem), and `<studyIDX>`. **Not** `<epochIDX>`.
- Rendered as `studyNNN_evaluation_<stem>.input` / `.<ext>`.
- The evaluator script (for example `eval_<harness>.py`) must be in
  `cp_files.txt`.
- Example: `applications/harnesses/ch_DDP_loderunner/` (`evaluation_input.tmpl`,
  `evaluation_slurm.tmpl`, `eval_LodeRunner.py`).
- Manual use: `yoke-evaluate-study --studyIDX N --checkpoint <path>
  [--dryrun]`.

## CLI flags

`yoke-start-study`: `--csv` (default `./hyperparameters.csv`), `--rundir`
(`./runs`), `--cpFile` (`./cp_files.txt`), `--submissionType`
(`slurm`|`shell`), `--dryrun`. Run it **from inside the harness directory**
(templates are read from `.`).

`yoke-evaluate-study`: `--studyIDX` (required), `--checkpoint` (required),
`--rundir`, `--csv`, `--cpFile`, `--submissionType`, `--dryrun`.
