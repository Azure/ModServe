#!/usr/bin/env bash
set -uo pipefail

# cluster manager, monolith/text public, vLLM internal, image nodes, transfer-recv, NCCL.
PORTS="7070 8080 8081 8001 8002 8086 8087 8088 8089 8090 8091 8092 8093 5051 5052 5053 5054 29500"

clean_local() {
    local killed=0 p pids pid pgid
    for p in $PORTS; do
        pids=$(ss -tlnpH "sport = :$p" 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u)
        for pid in $pids; do
            pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
            if [ -n "$pgid" ]; then kill -TERM -- "-$pgid" 2>/dev/null && killed=1; fi
            kill -TERM "$pid" 2>/dev/null && killed=1
        done
    done
    if [ "$killed" = 1 ]; then
        sleep 3
        for p in $PORTS; do
            pids=$(ss -tlnpH "sport = :$p" 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u)
            for pid in $pids; do
                pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
                if [ -n "$pgid" ]; then kill -KILL -- "-$pgid" 2>/dev/null; fi
                kill -KILL "$pid" 2>/dev/null
            done
        done
    fi
    echo "  cleanup $(hostname -s): swept campaign ports"
}

if [ "$#" -eq 0 ]; then
    clean_local
    exit 0
fi

SELF="$(hostname -I 2>/dev/null | tr ' ' '\n' | grep -E '^10\.' | head -1)"
SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/cluster_cleanup.sh"
for h in "$@"; do
    if [ "$h" = "$SELF" ] || [ "$h" = "127.0.0.1" ] || [ "$h" = "localhost" ]; then
        clean_local
    else
        ssh -i "$HOME/.ssh/id_cluster" -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new \
            "$h" "bash '$SCRIPT'" 2>/dev/null || echo "  cleanup $h: unreachable (skipped)"
    fi
done
exit 0
