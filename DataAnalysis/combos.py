# This script can be used to estimate how long it might take
# to grid search a certain number of columns/features.
# To run, set the num_features variable to the number of
# columns that are desired to be grid searched. This estimate
# assumes it takes one second to train and test a model.

from itertools import combinations

num_features = 14
test = range(num_features)
total = 0

for i in range(num_features):
    comb = combinations(test, i + 1)
    comb_list = list(comb)
    for j in comb_list:
        print(list(j))
    # print(comb_list)
    num_items = len(comb_list)
    total = total + num_items
    print(num_items)

print(f"Sum: {total}")
print(f"Hours to train @1sec per model: {total/60/60}")
