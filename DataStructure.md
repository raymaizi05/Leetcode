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
402. 移掉 K 位数字

```python
# 维护一个单调递增栈，来保留尽可能小的数字
class Solution:
    def removeKdigits(self, num: str, k: int) -> str:
        stack = []
        to_move = k
        for digit in num:
            while stack and to_move>0 and stack[-1]>digit:
                stack.pop()
                to_move-=1
            stack.append(digit)
        while to_move>0:
            stack.pop()
            to_move-=1
        
        result = "".join(stack).lstrip('0')

        return result if result!= "" else "0"
```
