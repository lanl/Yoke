# Saved Machine Profiles

One Markdown file per cluster, named `<clustername>.md` (lower case, from
`scontrol show config | grep ClusterName`). The `yoke-slurm-template` skill
creates and updates them, with the user's approval. Their format is in
`skills/yoke-slurm-template/reference/machine_profile_template.md`.

Machine facts here can be reused. Allocations listed here are dated hints
only, so re-query them every time.
