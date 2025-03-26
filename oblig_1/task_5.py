manyLists = [
    [8, 1, 10, 2, 9, 9, 8, 10, 4, 5],
    [8, 1, 4, 1, 1, 5, 2, 3, 10, 6],
    [7, 9, 1, 9, 5, 1, 1, 7, 8, 5],
    [2, 9, 10, 7, 7, 1, 6, 4, 6, 8],
    [7, 10, 4, 9, 6, 5, 4, 2, 6, 8],
    [8, 7, 3, 2, 6, 5, 4, 2, 9, 8],
    [6, 9, 1, 10, 7, 4, 7, 6, 10, 2],
    [2, 6, 2, 4, 3, 8, 3, 2, 1, 2],
    [6, 7, 4, 7, 1, 3, 4, 8, 10, 10],
    [6, 8, 3, 8, 4, 8, 3, 1, 10, 5],
]


def counter(someList):
    result = []
    for sublist in someList:
        count = 0
        for num in sublist:
            if num > 4:
                count += 1
        result.append(count)
    return print(result)


counter(manyLists)


# or this
def counters(someList):
    count = 0
    for num in someList:
        if num > 4:
            count += 1
    return count


counters(manyLists)
