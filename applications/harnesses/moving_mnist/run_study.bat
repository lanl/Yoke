@echo off
REM run_study.bat <train_script> <config_file>

REM 1) Ensure we’re in the study directory (we already are, thanks to cwd)
REM    so no pushd/popd at all!

REM 2) Point PYTHONPATH at the repo’s src folder
REM    %~dp0 is the folder where this .bat lives (the harness folder)
REM    so ../../.. takes us back to the root/src
set PYTHONPATH=%~dp0..\..\..\src

REM 3) Invoke Python on the copied harness (in study dir) with the input file
python %1 @%2
