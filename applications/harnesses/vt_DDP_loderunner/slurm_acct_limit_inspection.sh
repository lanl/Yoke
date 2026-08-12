#!/bin/bash

# Show your accounts
sacctmgr show user hickmank withass format="user,account%28,partition,qos%32"

# Inspect various slurm limits
scontrol show partition gpu | grep -E 'PartitionName|MaxTime|AllowQos|AllowAccounts'
sacctmgr show qos standard format=Name,MaxWall
sacctmgr show assoc where account=vt-normal_g user=$USER \
    format=Account,User,Partition,QOS,DefaultQOS,MaxWall
