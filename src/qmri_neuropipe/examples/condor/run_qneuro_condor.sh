#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 5 ]; then
  echo "Usage: $0 <subject> <session|none> <cpus> <memory_gb> <pipeline> [config_name] [require_dwi] [license_name]" >&2
  exit 2
fi

subject="${1#sub-}"
session="${2#ses-}"
session_lc="$(printf '%s' "$session" | tr '[:upper:]' '[:lower:]')"
cpus="$3"
memory_gb="$4"
pipeline="$5"
config_name="${6:-}"
require_dwi="${7:-true}"
license_name="${8:-license.txt}"

case "$session_lc" in
  ""|"none"|"null"|"n/a"|"na")
    session=""
    session_for_name="none"
    ;;
  *)
    session_for_name="$session"
    ;;
esac

scratch="${_CONDOR_SCRATCH_DIR:-$PWD}"
cd "$scratch"

echo "Running in Condor scratch directory: $PWD"
echo "Transferred top-level files and directories:"
find "$PWD" -maxdepth 2 -mindepth 1 -printf '  %P\n' | sort

if [ -f /.singularity.d/env/90-environment.sh ]; then
  # HTCondor's container integration should provide this environment already,
  # but sourcing it here makes PATH robust across site-specific launch modes.
  # shellcheck disable=SC1091
  source /.singularity.d/env/90-environment.sh || true
fi

export CONDA_DIR="${CONDA_DIR:-/opt/conda}"
export FSLDIR="${FSLDIR:-/usr/local/fsl}"
export FREESURFER_HOME="${FREESURFER_HOME:-/usr/local/freesurfer/8.2.0}"
export C3DPATH="${C3DPATH:-/opt/c3d/bin}"
export PATH="${CONDA_DIR}/bin:${FSLDIR}/bin:${FREESURFER_HOME}/bin:${FREESURFER_HOME}/python/bin:${FREESURFER_HOME}/python/scripts:${C3DPATH}:$PATH"

config=""
if [ -n "$config_name" ] && [ -f "$config_name" ]; then
  config="$PWD/$config_name"
else
  config="$(find "$PWD" -maxdepth 1 -type f \( -name '*.yaml' -o -name '*.yml' -o -name '*.json' \) | head -n 1)"
fi

if [ -z "$config" ]; then
  echo "No YAML/JSON qmri-neuropipe config was transferred into $PWD" >&2
  if [ -n "$config_name" ]; then
    echo "Expected config filename: $config_name" >&2
  fi
  echo "Check config_file in the submit file. It must be an absolute path readable from the submit host." >&2
  exit 4
fi

qmri_cmd=()
if command -v qmri-neuropipe >/dev/null 2>&1; then
  qmri_cmd=(qmri-neuropipe)
elif [ -x /opt/conda/bin/qmri-neuropipe ]; then
  qmri_cmd=(/opt/conda/bin/qmri-neuropipe)
elif [ -x /opt/conda/bin/python ] && /opt/conda/bin/python -c "import qmri_neuropipe.cli" >/dev/null 2>&1; then
  qmri_cmd=(/opt/conda/bin/python -m qmri_neuropipe.cli)
elif command -v python >/dev/null 2>&1 && python -c "import qmri_neuropipe.cli" >/dev/null 2>&1; then
  qmri_cmd=(python -m qmri_neuropipe.cli)
else
  echo "qmri-neuropipe was not found. This usually means the job did not start inside qneuro.sif or the container does not provide the installed package." >&2
  echo "PATH=$PATH" >&2
  echo "Container markers:" >&2
  find /.singularity.d /opt/conda/bin -maxdepth 1 -type f -o -type l 2>/dev/null | sed 's/^/  /' | head -n 50 >&2 || true
  exit 6
fi

mkdir -p data config out work
cp "$config" "config/$(basename "$config")"

license_path=""
if [ -n "$license_name" ] && [ -f "$license_name" ]; then
  license_path="$PWD/$license_name"
elif [ -f license.txt ]; then
  license_path="$PWD/license.txt"
else
  license_path="$(find "$PWD" -maxdepth 1 -type f -name '*license*.txt' | head -n 1)"
fi

if [ -n "$license_path" ]; then
  mkdir -p freesurfer
  cp "$license_path" freesurfer/license.txt
  export FS_LICENSE="$PWD/freesurfer/license.txt"
  echo "Using FreeSurfer license at FS_LICENSE=$FS_LICENSE"
else
  echo "Warning: no transferred FreeSurfer license found. FreeSurfer tools may fail unless the container/site provides FS_LICENSE." >&2
fi

# If the BIDS input was transferred as an archive, unpack it in scratch before
# reconstructing data/sub-*/ses-*. This is the most reliable pattern for
# HTCondor URL/staging plugins that do not recursively stage directories.
for archive in *.tar.gz *.tgz *.tar; do
  if [ -f "$archive" ] && [ "$archive" != "qneuro_outputs_sub-${subject}_ses-${session_for_name}.tar.gz" ]; then
    echo "Extracting transferred BIDS archive: $archive"
    tar -xf "$archive"
  fi
done

# Move any transferred BIDS root metadata into the staged BIDS root if present.
for root_file in dataset_description.json participants.tsv participants.json README CHANGES; do
  if [ -e "$root_file" ]; then
    mv "$root_file" data/
  fi
done

if [ -n "$session" ]; then
  if [ -d "sub-$subject" ]; then
    mv "sub-$subject" data/
  elif [ -d "ses-$session" ]; then
    # Condor transferred the session directory itself. Move it under the
    # expected BIDS subject root without creating data/sub-*/ses-*/ses-*.
    mkdir -p "data/sub-$subject"
    mv "ses-$session" "data/sub-$subject/"
  elif [ -d dwi ] || [ -d anat ] || [ -d fmap ] || [ -d func ] || [ -d perf ]; then
    # Some transfer modes can stage the contents of the session directory
    # directly into scratch. Re-wrap those modality folders as a BIDS session.
    mkdir -p "data/sub-$subject/ses-$session"
    for modality in anat dwi fmap func perf; do
      if [ -e "$modality" ]; then
        mv "$modality" "data/sub-$subject/ses-$session/"
      fi
    done
  else
    echo "Could not find transferred BIDS input sub-$subject/ses-$session or ses-$session" >&2
    find "$PWD" -maxdepth 2 -type d -print >&2
    exit 5
  fi
else
  if [ -d "sub-$subject" ]; then
    mv "sub-$subject" data/
  else
    echo "Could not find transferred BIDS input sub-$subject" >&2
    find "$PWD" -maxdepth 2 -type d -print >&2
    exit 5
  fi
fi

if [ ! -f data/dataset_description.json ]; then
  printf '{"Name":"condor-staged-qmri-neuropipe","BIDSVersion":"1.8.0"}\n' > data/dataset_description.json
fi

echo "Staged BIDS tree:"
find data -maxdepth 4 -type d -print | sort

echo "Staged BIDS files:"
find data -maxdepth 5 -type f -printf '  %P\n' | sort | head -n 200

if [ "$pipeline" = "dmri" ]; then
  dwi_dir="data/sub-$subject"
  if [ -n "$session" ]; then
    dwi_dir="$dwi_dir/ses-$session"
  fi
  dwi_dir="$dwi_dir/dwi"

  echo "DWI candidate files under $dwi_dir:"
  find "$dwi_dir" -maxdepth 1 -type f \( -name '*_dwi.nii.gz' -o -name '*_dwi.nii' -o -name '*_dwi.bval' -o -name '*_dwi.bvec' -o -name '*_dwi.json' \) -printf '  %P\n' 2>/dev/null | sort || true

  require_dwi_lc="$(printf '%s' "$require_dwi" | tr '[:upper:]' '[:lower:]')"
  if [ "$require_dwi_lc" != "false" ] && ! find "$dwi_dir" -maxdepth 1 -type f \( -name '*_dwi.nii.gz' -o -name '*_dwi.nii' \) 2>/dev/null | grep -q .; then
    echo "No BIDS DWI image found at $dwi_dir/*_dwi.nii[.gz]." >&2
    echo "Check bids_input in the submit file and confirm the source session contains a dwi/ directory with *_dwi.nii.gz, .bval, and .bvec files." >&2
    exit 7
  fi
fi

cmd=(
  "${qmri_cmd[@]}"
  --config "config/$(basename "$config")"
  --bids-dir data
  --output-dir out
  --work-dir work
  --pipeline "$pipeline"
  --participant-label "$subject"
  --n-cpus "$cpus"
  --memory-gb "$memory_gb"
)

if [ -n "$session" ]; then
  cmd+=(--session-label "$session")
fi

export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="$cpus"
export OMP_NUM_THREADS="$cpus"
export MKL_NUM_THREADS="$cpus"
export OPENBLAS_NUM_THREADS="$cpus"
export NUMEXPR_NUM_THREADS="$cpus"

"${cmd[@]}"

tarball="qneuro_outputs_sub-${subject}_ses-${session_for_name}.tar.gz"
tar -czf "$tarball" -C out .
echo "Wrote $tarball"
