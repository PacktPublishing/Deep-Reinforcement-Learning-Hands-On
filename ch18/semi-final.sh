#!/usr/bin/env bash
RES_DIR=saves/t18-more-mcts

extra=''

for round in `seq 0 9`; do
    echo Starting round $round
    if test $round -eq 9; then
        extra="$RES_DIR/best_1*"
    fi
    ./play.py --cuda -r 100 $RES_DIR/best_0${round}* $extra > tournament/semi-$round.txt &
done

echo "Waiting for them to finish"

for job in `jobs -p`; do
    echo $job
    wait $job || echo "Job $job failed"
done
