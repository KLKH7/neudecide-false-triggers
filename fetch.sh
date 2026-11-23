#!/bin/sh
# Fetch the public audio used by run.py into data/ (about 520 MB).
set -e
mkdir -p data && cd data
curl -sSL -o dev-clean-2.tar.gz https://www.openslr.org/resources/31/dev-clean-2.tar.gz
tar xzf dev-clean-2.tar.gz
python - <<'PY'
from huggingface_hub import hf_hub_download
hf_hub_download('macabdul9/IntentClassification_FluentSpeechCommands-Action', 'data/test-00000-of-00001.parquet',
                repo_type='dataset', local_dir='IntentClassification_FluentSpeechCommands-Action')
hf_hub_download('ashraq/esc50', 'data/train-00000-of-00002-2f1ab7b824ec751f.parquet',
                repo_type='dataset', local_dir='esc50')
PY
python - <<'PY'
from huggingface_hub import hf_hub_download
hf_hub_download('edinburghcstr/ami', 'sdm/validation-00004-of-00005.parquet', repo_type='dataset', local_dir='ami')
PY
mkdir -p silero
curl -sSL -o silero/silero_vad.onnx https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx
python - <<'PY'
from huggingface_hub import hf_hub_download
hf_hub_download('edinburghcstr/ami', 'ihm/validation-00004-of-00005.parquet', repo_type='dataset', local_dir='ami')
PY
python - <<'PY'
from huggingface_hub import hf_hub_download
hf_hub_download('distil-whisper/meanwhile', 'data/test-00000-of-00001-313dec7dc01b89d3.parquet', repo_type='dataset', local_dir='meanwhile')
PY
