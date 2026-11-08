import time
with open('voting_record.txt', 'r') as file:
    count = 0
    for line in file:
        values = line.strip().split()
        for value in values:
            if float(value)!=0 and float(value)!=1:
                continue
            if float(value) == 1.0:
                count += 1
    print(f"The number of 1's in the file is: {count}")