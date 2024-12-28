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
如果是一个list里有1-10的数字，去除K个元素之后求可能的最大值

```python
def max_number_after_remove(nums, K):
    """
    根据题意所说方法：
    1) '优先选10'
    2) 再从左到右做单调栈
    3) 如果还有剩余要删的，再从右到左删除
    返回拼接后的最大数字（字符串形式）。
    """
    stack = []
    remain = K
    
    for x in nums:
        # 如果是 10，直接压栈（不弹栈）
        if x == 10:
            stack.append(x)
        else:
            # 如果不是 10，才做单调栈比较
            while stack and stack[-1] != 10 and remain > 0 and stack[-1] < x:
                stack.pop()
                remain -= 1
            stack.append(x)
    print(stack)
    # 如果还没删够，就从右往左再删
    # 这里的策略是：从右往左，如果某个位置的数字比右边小，就删掉它
    # 一直到删够 remain == 0 或者扫不动了
    i = len(stack) - 1
    while remain > 0 and i >= 0:
        if stack[i] < stack[i-1]:
            stack.pop(i)
            remain -= 1
            # 删掉第 i 个之后，要向左退一步，继续比较
            i -= 1
        else:
            i -= 1

    # 如果还剩要删，但右->左也删不动了，就从末尾再删掉
    while remain > 0 and stack:
        stack.pop()
        remain -= 1

    # 拼接结果
    return ''.join(str(x) for x in stack)
```
