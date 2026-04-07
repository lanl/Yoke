YOKE: Yielding Optimal Knowledge Enhancement
============================================

[![Coverage Status](https://coveralls.io/repos/github/lanl/Yoke/badge.svg?branch=main)](https://coveralls.io/github/lanl/Yoke?branch=main)
[![pipeline status](https://github.com/lanl/Yoke/actions/workflows/yoke_install_test_lint.yml/badge.svg)](https://github.com/lanl/Yoke/actions) 
[![Latest Release](https://img.shields.io/github/v/release/lanl/Yoke)](https://github.com/lanl/Yoke/releases)

![Get YOKEd!](./YOKE_DALLE_512x512.png)

About
-----

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
- utils/training/
- utils/training/datastep/
- utils/training/epoch/
- lr_schedulers.py
- parellel_utils.py

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

Installation
------------

The Python environment is specified through the `pyproject.toml`
file. YOKE is meant to be installed using `flit` in a minimal python
environment. We recommend a vanilla python virtual environment.

Setup your base environment and activate it:

```bash
>> python -m venv $HOME/<yoke_env_name> python=3.11
>> source $HOME/<yoke_env_name>/bin/activate
>> pip install flit
>> cd <yoke_repo_clone_directory>
>> flit install
```

Alternatively, one can do a developer install, which allows editing in the `yoke` clone
and includes testing dependencies, with

```bash
>> flit install --symlink --deps develop
```

An alternative to building your Python environment for a particular computing
environment is to use containerization. This allows for portability and some
degree of isolation from the host system. You can then develop a Yoke
application on one system and easily move it to another for larger training
runs etc.

Using the Docker runtime, build the container (this could be done as part of a
CI pipeline) and drop in to an interative shell:

```bash
>> cd <yoke_repo_clone_directory>
>> docker buildx build -f Dockerfile . -t yoke
>> docker run -it yoke /bin/bash
```

When you want to push your application to a different machine or deploy to a
service like Kubernetes, RunAI etc., you can run `docker push` or
`docker export` along with any transfers of necessary training data.

Testing
-------

To run the tests use ...

> **NOTE**
> A developer install is recommended for testing. Otherwise you'll have to also install
> `pytest`, `pytest-cov`, `ruff`, and `coverage` in your python environment separately.

```bash
>> pytest
>> pytest --cov
>> pytest --cov --cov-report term-missing
```

Linting
-------

The `ruff` linter is used in `YOKE` to enforce coding and formatting
standards. To run the linter do

```bash
>> ruff check
>> ruff check --preview
```

You can make `ruff` fix automatic standards using

```bash
>> ruff check --fix
>> ruff check --preview --fix
```

Use `ruff` to then check your code formatting and show you what would
be adjusted, then fix formatting

```bash
>> ruff format --check --diff
>> ruff format
```

Copyright
---------

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
