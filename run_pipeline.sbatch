#!/usr/bin/env bash
#SBATCH --job-name=darna-pipeline
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

PROJ=/mnt/beegfs/projects/CSCI451_26F/jp258213/bio_project
cd "$PROJ"
mkdir -p trimmed alignments results logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate darna

THREADS="${SLURM_CPUS_PER_TASK:-8}"
IDX="$PROJ/reference/NC_000913.2"
AD="$PROJ/darna-trna/config/adapters.fa"
SRC="$PROJ/darna-trna/src"

echo "job      ${SLURM_JOB_ID:-local}"
echo "node     $(hostname)"
echo "threads  $THREADS"
echo "started  $(date -Iseconds)"

# SRR37695883 = GSM9603520 = DARNA plasmid, T7-infected   (library 3204)
# SRR37695884 = GSM9603519 = empty plasmid, T7-infected   (library 3203)
run () {
  local label="$1" acc="$2"
  echo
  echo "=== $label ($acc) ==="

  echo "[$label] trimming"
  # Flags exactly as the paper's methods state. tbo uses mate overlap to find
  # the insert boundary, which is the strongest signal available when the
  # fragment is shorter than the read -- the common case here.
  bbduk.sh -Xmx16g \
    in1="raw/${acc}_1.fastq.gz" in2="raw/${acc}_2.fastq.gz" \
    out1="trimmed/${label}_R1.fastq.gz" out2="trimmed/${label}_R2.fastq.gz" \
    ref="$AD" ktrim=r k=20 mink=10 hdist=1 tpe tbo minlen=17 \
    threads="$THREADS" overwrite=t \
    2> "logs/${label}_bbduk.log"
  grep -E "^(Input|KTrimmed|Total Removed|Result):" "logs/${label}_bbduk.log" || true

  echo "[$label] aligning and counting"
  # Piped rather than writing SAM to disk: ~20 GB per sample otherwise, and
  # nothing downstream needs the alignments as a file.
  bowtie2 -x "$IDX" \
      -1 "trimmed/${label}_R1.fastq.gz" -2 "trimmed/${label}_R2.fastq.gz" \
      --end-to-end -p "$THREADS" \
      2> "logs/${label}_bowtie2.log" \
    | python3 -I "$SRC/count_ends.py" - \
        -o "results/${label}_counts.npz" --end-mode fragment \
        2> "logs/${label}_count.log"

  echo "--- bowtie2 ---"; tail -6 "logs/${label}_bowtie2.log"
  echo "--- counts ---";  cat "logs/${label}_count.log"
}

run darna   SRR37695883
run control SRR37695884

echo
echo "=== comparing ==="
python3 -I "$SRC/analyze.py" \
  --treat results/darna_counts.npz \
  --ctrl  results/control_counts.npz \
  --out   results/signals.npz \
  --table results/ranked_positions.tsv \
  --min-count 50 --top 30

echo
echo "finished $(date -Iseconds)"
sacct -j "${SLURM_JOB_ID:-0}" --format=JobID,Elapsed,MaxRSS,State 2>/dev/null || true
