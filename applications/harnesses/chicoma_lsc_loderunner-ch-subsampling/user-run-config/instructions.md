# Steps to run channel subsampling study runs

## Step 1:
### set YOKE_ROOT to the top-level dir for your yoke fork on chicoma.
#### You will need to change the following line.
YOKE_ROOT="$HOME/temp/ch-subsampling-tests"


### Set env var to point to the chicoma channel subsampling dir
CH_SS_HARNESS=${YOKE_ROOT}/applications/harnesses/chicoma_lsc_loderunner-ch-subsampling

## Step 2: Make sure that your git repo is up-to-date with the main git repo
        cd $YOKE_ROOT

### Add main git repo as remote. If you have alredy added this remote with some
### other name, replace 'yoke_team' with the name you are using.
        git add remote yoke_team https://github.com/lanl/Yoke.git

### Bring changes from team's yoke repo
        git fetch yoke_team

### Merge changes from yoke_team into your current branch
#### If there are conflicts, your would need to resolve them
        git merge yoke_team/main

## Step 3: Create Symbolin Link 'runs'.

        cd ${CH_SS_HARNESS}
        ln -sf /lustre/scratch5/exempt/artimis/mpmm/spandit/runs_yoke_chicoma ./runs

## Step 4: Copy user specific files to familiar config files for yoke
        yes y | cp -f user-run-config/hyperparameters.csv.$USER  hyperparameters.csv
        yes y | cp -r user-run-config/training_START.input.$USER training_START.input
        yes y | cp -f user-run-config/training_input.tmpl.$USER  training_input.tmpl

## Step 5: Start the studies"
        cd ${CH_SS_HARNESS}
        python3 START_study.py



