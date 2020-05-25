# https://en.wikipedia.org/wiki/Moore_neighborhood
# moore(0) -> 1x1 square
# moore(1) -> 3x3 square
# moore(2) -> 5x5 square
# ...
def moore(n):
    length = 2 * n + 1
    neighborhood = []

    for i in range(length):
        neighborhood.append([])
        for _ in range(length):
            neighborhood[i].append(True)

    return neighborhood


# https://en.wikipedia.org/wiki/Von_Neumann_neighborhood
# "Manhattan distance"
def von_neumann(n):
    length = 2 * n + 1
    center = length // 2
    neighborhood = []

    for i in range(length):
        neighborhood.append([])
        for j in range(length):
            neighborhood[i].append(abs(i - center) + abs(j - center) <= center)

    return neighborhood
