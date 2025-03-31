YOKE: Yielding Optimal Knowledge Enhancement
============================================

[![Coverage Status](https://coveralls.io/repos/github/lanl/Yoke/badge.svg?branch=main)](https://coveralls.io/github/lanl/Yoke?branch=main)
[![pipeline status](https://github.com/lanl/Yoke/actions/workflows/yoke_install_test_lint.yml/badge.svg)](https://github.com/lanl/Yoke/actions) 
[![Latest Release](https://img.shields.io/github/v/release/lanl/Yoke)](https://github.com/lanl/Yoke/releases)

![Get YOKEd!](./YOKE_DALLE_512x512.png)

About:
------

A general prototyping, training, and testing harness for pytorch used
for models developed under the **ArtIMis: Multi-physics/Multi-material
Applications** and **ASC-PEM-EADA(Enabling Agile Design and Assessment)**
projects.

The YOKE module is divided into submodules, installed in a python environment:

- datasets/
- helpers/
- models/
- metrics/
- losses/
- utils/
- lr_schedulers.py
- parellel_utils.py
- torch_training_utils.py

Helper utilities and examples are under `applications`:

- harnesses
- makefilelists.py
- filelists
- normalization
- evaluation
- viewers

NOTE: Data for training is not housed within YOKE. The data locations are
specified through command-line arguments passed to the programs in
`harnesses`, `evaluation`, and `viewers`.


Installation:
-------------

The python environment is specified through the `pyproject.toml`
file. YOKE is meant to be installed using `flit` in a minimal python
environment.

Setup your base environment and activate it (we use conda):

```
>> conda create -n <yoke_env_name> python=3.9 flit
>> conda activate <yoke_env_name>
>> cd <yoke_repo_clone_directory>
>> flit install
```

> **WARNING**
> 
> This install process does not guarantee that PyTorch is installed to
> utilize your GPUs. If you want to ensure that PyTorch is installed to
> make optimal use of your hardware we suggest manually installing
> `torch` prior to installing `YOKE` with `flit`.

Testing:
--------

To run the tests use...

```
>> pytest
>> pytest --cov
>> pytest --cov --cov-report term-missing
```

Linting:
--------

The `ruff` linter is used in `YOKE` to enforce coding and formatting
standards. To run the linter do

```
>> ruff check
>> ruff check --preview
```

You can make `ruff` fix automatic standards using

```
>> ruff check --fix
>> ruff check --preview --fix
```

Use `ruff` to then check your code formatting and show you what would
be adjusted, then fix formatting

```
>> ruff format --check --diff
>> ruff format
```

Copyright:
----------

LANL **O4863**

&copy; 2025. Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract 89233218CNA000001 for Los
Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for
the U.S. Department of Energy/National Nuclear Security Administration. All rights in
the program are reserved by Triad National Security, LLC, and the U.S. Department of
Energy/National Nuclear Security Administration. The Government is granted for itself
and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license
in this material to reproduce, prepare. derivative works, distribute copies to the
public, perform publicly and display publicly, and to permit others to do so.
