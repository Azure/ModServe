#!/usr/bin/env bash
# Push this repo to remote Azure DGX A100 boxes and build the env on each.
#   utils/remote_setup.sh <host> [<host> ...]      # host = [user@]hostname
# Needs key-based SSH from this box to each remote. Safe to re-run.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ "$#" -gt 0 ] || { echo "usage: $0 <host> [<host> ...]" >&2; exit 2; }

SSH="ssh -o StrictHostKeyChecking=accept-new"
rc=0
for host in "$@"; do
    echo "==== $host ===="
    if rsync -az --delete -e "$SSH" \
        --exclude '.git/' --exclude '/data/' --exclude '/metrics/' --exclude '/logs/' \
        --exclude '/results/' --exclude '__pycache__/' --exclude '*.pyc' \
        "$REPO_ROOT/" "$host:lmm-serving/" \
        && $SSH "$host" 'cd lmm-serving && bash utils/setup_repo.sh'; then
        echo "  OK: $host ready"
    else
        echo "  FAIL: $host" >&2; rc=1
    fi
done
exit $rc
