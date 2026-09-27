#!/bin/bash
# package.sh <name> <matching_results.tsv> <candidate_pairs.tsv>
# Builds "/Volumes/T7 Shield/KRISH/AMAZON HACKATHON/submissions/<name>/" in the official zip layout
#   output/{matching_results.tsv,candidate_pairs.tsv}, code/business_entity_resolution/{src,configs,README.md,requirements.txt},
#   Documentation_template.md
# runs utils/validate_submission.py --check-ids on it, and zips it to submissions/<name>_submission.zip
# (no "._*" stubs, no __pycache__). Never touches output/ or any source file.
set -eu
[ $# -eq 3 ] || { echo "usage: package.sh <name> <matching.tsv> <candidates.tsv>"; exit 2; }
REPO="/Volumes/T7 Shield/KRISH/AMAZON HACKATHON/amazon-ml-challenge"
DEST="/Volumes/T7 Shield/KRISH/AMAZON HACKATHON/submissions"
NAME=$1; M=$2; C=$3
D="$DEST/$NAME"
rm -rf "$D" "$DEST/${NAME}_submission.zip"
mkdir -p "$D/output" "$D/code/business_entity_resolution"
cp "$M" "$D/output/matching_results.tsv"
cp "$C" "$D/output/candidate_pairs.tsv"
CB="$REPO/code/business_entity_resolution"
rsync -a --exclude '._*' --exclude '__pycache__' "$CB/src" "$CB/configs" "$D/code/business_entity_resolution/"
cp "$CB/README.md" "$CB/requirements.txt" "$D/code/business_entity_resolution/"
cp "$REPO/Documentation_template.md" "$D/"
cd "$REPO"
"$REPO/.venv/bin/python" utils/validate_submission.py --matching "$D/output/matching_results.tsv" \
  --candidate "$D/output/candidate_pairs.tsv" --test-dir dataset/test --check-ids | tee "$DEST/${NAME}_validate.txt"
cd "$D" && find . -name '._*' -delete && COPYFILE_DISABLE=1 zip -qr -X "$DEST/${NAME}_submission.zip" . -x '*/._*' '._*'
echo "ZIP $DEST/${NAME}_submission.zip ($(du -h "$DEST/${NAME}_submission.zip" | cut -f1))"
