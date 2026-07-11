#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
Usage:
  package_and_submit_qneuro.sh --submit-file qneuro_single_subject.sub \
    --bids-dir /path/to/bids --subject 10021 --session 01 \
    --package-dir /path/to/package-or-staging-dir [--transfer-uri osdf:///chtc/staging/user]

  package_and_submit_qneuro.sh --submit-file qneuro_single_subject.sub \
    --bids-dir /path/to/bids --subjects-file subjects.csv \
    --package-dir /path/to/package-or-staging-dir [--transfer-uri osdf:///chtc/staging/user]

  package_and_submit_qneuro.sh --submit-file qneuro_single_subject.sub \
    --remote-bids user@remote.host:/path/to/bids --subjects-file subjects.csv \
    --package-dir /chtc/staging/dir [--transfer-uri osdf:///chtc/staging/user]

Creates one BIDS archive per subject/session containing:
  - sub-<subject>/ses-<session>, or sub-<subject> for no-session datasets
  - root BIDS metadata files when present

Then writes a temporary submit file that queues one job per packaged row and
submits it with condor_submit. subjects.csv should contain:
  subject,session

If --transfer-uri is provided, tarballs are created in --package-dir but the
submit file uses:
  <transfer-uri>/<tarball-name>

Example:
  package_and_submit_qneuro.sh \
    --submit-file qneuro.sub \
    --remote-bids deaniii@wbic.example.edu:/data/bids \
    --subjects-file subjects.csv \
    --package-dir /home/deaniii/wbic/staging \
    --transfer-uri osdf:///chtc/staging/deaniii
USAGE
}

submit_file=""
bids_dir=""
remote_bids=""
remote_host=""
remote_bids_dir=""
subject=""
session=""
subjects_file=""
package_dir=""
transfer_uri=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --submit-file)
      submit_file="${2:-}"
      shift 2
      ;;
    --bids-dir)
      bids_dir="${2:-}"
      shift 2
      ;;
    --remote-bids)
      remote_bids="${2:-}"
      shift 2
      ;;
    --subject)
      subject="${2:-}"
      shift 2
      ;;
    --session)
      session="${2:-}"
      shift 2
      ;;
    --subjects-file)
      subjects_file="${2:-}"
      shift 2
      ;;
    --package-dir)
      package_dir="${2:-}"
      shift 2
      ;;
    --transfer-uri)
      transfer_uri="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [ -z "$submit_file" ] || [ -z "$package_dir" ]; then
  usage
  exit 2
fi

if [ -z "$bids_dir" ] && [ -z "$remote_bids" ]; then
  echo "Provide either --bids-dir or --remote-bids." >&2
  usage
  exit 2
fi

if [ -n "$bids_dir" ] && [ -n "$remote_bids" ]; then
  echo "Use either --bids-dir or --remote-bids, not both." >&2
  exit 2
fi

if [ -z "$subjects_file" ] && [ -z "$subject" ]; then
  echo "Provide either --subject or --subjects-file." >&2
  usage
  exit 2
fi

if [ -n "$subjects_file" ] && [ -n "$subject" ]; then
  echo "Use either --subject/--session or --subjects-file, not both." >&2
  exit 2
fi

if [ ! -f "$submit_file" ]; then
  echo "Submit file not found: $submit_file" >&2
  exit 3
fi

submit_dir="$(cd "$(dirname "$submit_file")" && pwd -P)"
submit_name="$(basename "$submit_file")"
submit_file="$submit_dir/$submit_name"

if [ -n "$bids_dir" ] && [ ! -d "$bids_dir" ]; then
  echo "BIDS directory not found: $bids_dir" >&2
  exit 3
fi

if [ -n "$remote_bids" ]; then
  if [[ "$remote_bids" != *:* ]]; then
    echo "Remote BIDS path must look like user@host:/path/to/bids" >&2
    exit 3
  fi
  remote_host="${remote_bids%%:*}"
  remote_bids_dir="${remote_bids#*:}"
  if [ -z "$remote_host" ] || [ -z "$remote_bids_dir" ]; then
    echo "Remote BIDS path must look like user@host:/path/to/bids" >&2
    exit 3
  fi
fi

if [ -n "$subjects_file" ] && [ ! -f "$subjects_file" ]; then
  echo "Subjects file not found: $subjects_file" >&2
  exit 3
fi

mkdir -p "$package_dir"
tmp_submit="$(mktemp "$submit_dir/.qneuro_submit.XXXXXX.sub")"
vars_file="$(mktemp "$submit_dir/.qneuro_inputs.XXXXXX.csv")"

if [ -n "$transfer_uri" ]; then
  transfer_uri="${transfer_uri%/}"
fi

package_pair() {
  local pair_subject="$1"
  local pair_session="$2"
  local subject_dir session_dir input_rel archive_name archive_path bids_input
  local tar_inputs root_file

  pair_subject="${pair_subject#sub-}"
  pair_session="${pair_session#ses-}"

  if [ -z "$pair_subject" ]; then
    return 0
  fi

  subject_dir="sub-$pair_subject"

  if [ -n "$pair_session" ]; then
    session_dir="ses-$pair_session"
    input_rel="$subject_dir/$session_dir"
    archive_name="${subject_dir}_${session_dir}_bids.tar.gz"
  else
    input_rel="$subject_dir"
    archive_name="${subject_dir}_bids.tar.gz"
  fi

  archive_path="$package_dir/$archive_name"
  tar_inputs=("$input_rel")

  if [ -n "$remote_bids" ]; then
    echo "Checking remote BIDS input: $remote_host:$remote_bids_dir/$input_rel"
    if ! ssh "$remote_host" test -d "$(printf '%q' "$remote_bids_dir/$input_rel")"; then
      echo "Requested remote BIDS input not found: $remote_host:$remote_bids_dir/$input_rel" >&2
      exit 4
    fi

    for root_file in dataset_description.json participants.tsv participants.json README CHANGES; do
      if ssh "$remote_host" test -e "$(printf '%q' "$remote_bids_dir/$root_file")"; then
        tar_inputs+=("$root_file")
      fi
    done

    echo "Streaming remote BIDS archive to: $archive_path"
    remote_tar_args=""
    for item in "${tar_inputs[@]}"; do
      remote_tar_args+=" $(printf '%q' "$item")"
    done
    ssh "$remote_host" "tar -C $(printf '%q' "$remote_bids_dir") -czf -$remote_tar_args" > "$archive_path"
  else
    if [ ! -d "$bids_dir/$input_rel" ]; then
      echo "Requested BIDS input not found: $bids_dir/$input_rel" >&2
      exit 4
    fi

    for root_file in dataset_description.json participants.tsv participants.json README CHANGES; do
      if [ -e "$bids_dir/$root_file" ]; then
        tar_inputs+=("$root_file")
      fi
    done

    echo "Creating BIDS archive: $archive_path"
    tar -C "$bids_dir" -czf "$archive_path" "${tar_inputs[@]}"
  fi

  if [ -n "$transfer_uri" ]; then
    bids_input="$transfer_uri/$archive_name"
  else
    bids_input="$archive_path"
  fi

  # Queue both uppercase and lowercase macro names so this works with submit
  # files that use either $(SUBJECT)/$(SESSION) or $(subject)/$(session).
  printf '%s,%s,%s,%s,%s\n' "$pair_subject" "$pair_session" "$bids_input" "$pair_subject" "$pair_session" >> "$vars_file"
}

if [ -n "$subjects_file" ]; then
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line%%#*}"
    line="$(printf '%s' "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    if [ -z "$line" ]; then
      continue
    fi

    IFS=',' read -r row_subject row_session _extra <<< "$line"
    row_subject="$(printf '%s' "${row_subject:-}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    row_session="$(printf '%s' "${row_session:-}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    case "$(printf '%s' "$row_subject" | tr '[:upper:]' '[:lower:]')" in
      subject|sub|participant|participant_label)
        continue
        ;;
    esac
    package_pair "$row_subject" "$row_session"
  done < "$subjects_file"
else
  package_pair "$subject" "$session"
fi

if [ ! -s "$vars_file" ]; then
  echo "No subject/session rows were packaged." >&2
  exit 4
fi

awk \
  -v vars_file="$(basename "$vars_file")" \
  '
    BEGIN {
      wrote_queue = 0
    }
    /^[[:space:]]*SUBJECT[[:space:]]*=/ { next }
    /^[[:space:]]*subject[[:space:]]*=/ { next }
    /^[[:space:]]*SESSION[[:space:]]*=/ { next }
    /^[[:space:]]*session[[:space:]]*=/ { next }
    /^[[:space:]]*bids_input[[:space:]]*=/ { next }
    /^[[:space:]]*queue[[:space:]]/ || /^[[:space:]]*QUEUE[[:space:]]/ {
      if (!wrote_queue) {
        print "queue SUBJECT,SESSION,bids_input,subject,session from " vars_file
        wrote_queue = 1
      }
      next
    }
    { print }
    END {
      if (!wrote_queue) {
        print "queue SUBJECT,SESSION,bids_input,subject,session from " vars_file
      }
    }
  ' "$submit_file" > "$tmp_submit"

echo "Submitting queued inputs from: $vars_file"
(
  cd "$submit_dir"
  condor_submit "$(basename "$tmp_submit")"
)
echo "Temporary submit file: $tmp_submit"
echo "Queue vars file: $vars_file"
