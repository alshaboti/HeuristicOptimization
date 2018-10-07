import itertools

lst = [1, 2, 3]
combs = []

for i in range(1, len(lst)+1):
    combs.append(i)
    els = [list(x) for x in itertools.combinations(lst, i)]
    combs.append(els)
print()
