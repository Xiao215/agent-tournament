#!/bin/bash

YEAR=$(date +%Y)
MONTH=$(date +%m)
DAY=$(date +%d)
OUTDIR="sbatch-output/$YEAR/$MONTH/$DAY"
mkdir -p "$OUTDIR"

JOBNAME="IPD_sim"
sbatch --job-name="$JOBNAME" \
       --output="$OUTDIR/${JOBNAME}-%j.out" \
       --error="$OUTDIR/${JOBNAME}-%j.err" \
       run_job.sh