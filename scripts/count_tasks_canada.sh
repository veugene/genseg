#!/bin/bash
TMP_SACCT_OUT=`sacct -u veugene`
N_PENDING=$(grep PENDING <<< "$TMP_SACCT_OUT" |wc -l)
N_RUNNING=$(expr $(grep RUNNING <<< "$TMP_SACCT_OUT" |wc -l) / 2)
echo "PENDING: $N_PENDING  RUNNING: $N_RUNNING"
