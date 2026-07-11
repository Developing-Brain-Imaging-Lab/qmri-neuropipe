#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
Usage:
  push_package_and_submit_qneuro.sh --submit-host user@chtc.submit.host \
    --remote-submit-dir /home/user/condor/qneuro \
    --remote-package-dir /home/user/staging \
    --transfer-uri osdf:///chtc/staging/user \
    --submit-file qneuro.sub --bids-dir /path/to/bids \
    --subjects-file subjects.csv

Run this on a BIDS server that can SSH to the CHTC submit host when the CHTC
submit host cannot SSH back to the BIDS server.

It will:
  1. Create one BIDS tarball per subject/session locally.
  2. scp those tarballs to --submit-host:--remote-package-dir.
  3. Generate a temporary submit file plus queue vars file locally.
  4. scp both files to --submit-host:--remote-submit-dir.
  5. Run condor_submit on the CHTC submit host over SSH.

subjects.csv should contain:
  subject,session

The submit file should use $(bids_input) in transfer_input_files.
USAGE
}

submit_host=""
remote_submit_dir=""
remote_package_dir=""
transfer_uri=""
submit_file=""
bids_dir=""
subject=""
session=""
subjects_file=""
copy_executable="false"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --submit-host)
      submit_host="${2:-}"
      shift 2
      ;;
    --remote-submit-dir)
      remote_submit_dir="${2:-}"
      shift 2
      ;;
    --remote-package-dir)
      remote_package_dir="${2:-}"
      shift 2
      ;;
    --transfer-uri)
      transfer_uri="${2:-}"
      shift 2
      ;;
    --submit-file)
      submit_file="${2:-}"
      shift 2
      ;;
    --bids-dir)
      bids_dir="${2:-}"
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
    --copy-executable)
      copy_executable="true"
      shift
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

if [ -z "$submit_host" ] || [ -z "$remote_submit_dir" ] || [ -z "$remote_package_dir" ] || [ -z "$transfer_uri" ] || [ -z "$submit_file" ] || [ -z "$bids_dir" ]; then
  usage
  exit 2
fi

if [ -z "$subjects_file" ] && [ -z "$subject" ]; then
  echo "Provide either --subject or --subjects-file." >&2
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

if [ ! -d "$bids_dir" ]; then
  echo "BIDS directory not found: $bids_dir" >&2
  exit 3
fi

if [ -n "$subjects_file" ] && [ ! -f "$subjects_file" ]; then
  echo "Subjects file not found: $subjects_file" >&2
  exit 3
fi

work_dir="$(mktemp -d "${TMPDIR:-/tmp}/qneuro_push.XXXXXX")"
archive_dir="$work_dir/archives"
mkdir -p "$archive_dir"

submit_base="$(basename "$submit_file")"
tmp_submit="$work_dir/$submit_base"
vars_file="$work_dir/qneuro_inputs.csv"
transfer_uri="${transfer_uri%/}"

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

  if [ ! -d "$bids_dir/$input_rel" ]; then
    echo "Requested BIDS input not found: $bids_dir/$input_rel" >&2
    exit 4
  fi

  archive_path="$archive_dir/$archive_name"
  tar_inputs=("$input_rel")
  for root_file in dataset_description.json participants.tsv participants.json README CHANGES; do
    if [ -e "$bids_dir/$root_file" ]; then
      tar_inputs+=("$root_file")
    fi
  done

  echo "Creating BIDS archive: $archive_path"
  tar -C "$bids_dir" -czf "$archive_path" "${tar_inputs[@]}"
  bids_input="$transfer_uri/$archive_name"

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
    BEGIN { wrote_queue = 0 }
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

echo "Creating remote directories on $submit_host"
ssh "$submit_host" "mkdir -p $(printf '%q' "$remote_submit_dir") $(printf '%q' "$remote_package_dir")"

echo "Copying BIDS archives to $submit_host:$remote_package_dir"
scp "$archive_dir"/*.tar.gz "$submit_host:$remote_package_dir/"

echo "Copying submit files to $submit_host:$remote_submit_dir"
scp "$tmp_submit" "$vars_file" "$submit_host:$remote_submit_dir/"

if [ "$copy_executable" = "true" ]; then
  executable_path="$(awk -F= 'tolower($1) ~ /^[[:space:]]*executable[[:space:]]*$/ {gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2); print $2; exit}' "$submit_file")"
  if [ -n "$executable_path" ] && [ -f "$(dirname "$submit_file")/$executable_path" ]; then
    echo "Copying executable $executable_path to $submit_host:$remote_submit_dir"
    scp "$(dirname "$submit_file")/$executable_path" "$submit_host:$remote_submit_dir/"
  fi
fi

echo "Submitting on $submit_host"
ssh "$submit_host" "cd $(printf '%q' "$remote_submit_dir") && condor_submit $(printf '%q' "$submit_base")"
echo "Local staging work directory: $work_dir"
