Queue 

Given array and fixed window , return the rolling max. O(n)


```python
from collections import deque
def sliding_window_max(nums, k):
    #Maintain a decreasing deque
    if not nums or k <= 0:
        return []

    n = len(nums)
    max_indices = []  # 模拟 deque 的索引数组
    result = []

    for i in range(n):
        # 移除超出窗口的索引
        if max_indices and max_indices[0] < i - k + 1:
            max_indices.pop(0)

        # 移除队列中比当前值小的所有索引
        while max_indices and nums[max_indices[-1]] < nums[i]:
            max_indices.pop()

        # 将当前索引加入队列
        max_indices.append(i)

        result.append(nums[max_indices[0]])

    return result

```
