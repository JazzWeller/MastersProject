#!/bin/bash
# Pauses every long-running training / evaluation job in this WSL instance
# (ml/resume.py). From Windows:
#
#     wsl -- bash /mnt/c/Users/Jazz/Documents/kf-observation/Code/Non-GUI/tools/pause_training.sh
#
# Each job gets SIGTERM: a BC trainer (ml.bc_train, ml.bc_train_v2 and the
# screen tools that call them) saves its state at the next step and exits
# with code 75; the self-play learner saves and stops its actors; the search
# evaluation (tools.eval_v2) keeps the seed units it has finished. Rerun the
# same command to resume. Waits (up to --wait seconds, default 600) until
# they have all exited, so the machine is free when this returns.

wait=600
[ "$1" = "--wait" ] && wait=$2
pattern='-m (ml\.bc_train|ml\.bc_train_v2|ml\.selfplay_train|tools\.screen5_v2|tools\.run_screens|tools\.deck_generalization|tools\.eval_v2)( |$)'
pids=""
for p in $(pgrep -f -- "$pattern"); do  # the interpreters only, not shells that launched them
    case "$(cat /proc/$p/comm 2>/dev/null)" in python*|pypy*) pids="$pids $p" ;; esac
done
if [ -z "$pids" ]; then
    echo "nothing to pause"
    exit 0
fi
for p in $pids; do
    echo "pausing $p: $(tr '\0' ' ' < /proc/$p/cmdline | cut -c1-160)"
    kill -TERM "$p"
done
for _ in $(seq "$wait"); do
    alive=""
    for p in $pids; do kill -0 "$p" 2>/dev/null && alive="$alive $p"; done
    [ -z "$alive" ] && { echo "all paused"; exit 0; }
    sleep 1
done
echo "still running after ${wait}s:$alive"
exit 1
