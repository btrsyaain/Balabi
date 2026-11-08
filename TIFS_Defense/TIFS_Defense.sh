for label in 0 1 2 3 4 5 6 7 8 9 
    do 
        python3 poison_train_defense.py --dup 0 --label ${label} >> log.txt 2>&1
    done