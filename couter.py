
from collections import Counter

def topKFrequent(nums: list[int], k: int) -> list[int]:
    freq = Counter(nums)

    buckets = [[] for _ in range(len(nums) + 1)]

    for num, count in freq.items():
        buckets[count].append(num)

    ans = []

    for count in range(len(nums), 0, -1):
        for num in buckets[count]:
            ans.append(num)

            if len(ans) == k:
                return ans

    return ans
