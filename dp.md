## [300. Longest Increasing Subsequence](https://leetcode.com/problems/longest-increasing-subsequence/)

```python

    def lengthOfLIS(self, nums: List[int]) -> int:
        # dp+bs Time Complexity O(nlogn)
        # dp[i[ is smallest last number of increasing sequence with length i+1
        def bisect_left(arr, x):
            left, right = 0, len(arr)
            while left < right:
                mid = (left + right) // 2
                if arr[mid] < x:
                    left = mid + 1
                else:
                    right = mid
            return left
        dp = []
        for num in nums:
            if not dp or num>dp[-1]:
                dp.append(num)
            else:
                loc = bisect_left(dp, num)
                dp[loc] = num
        return len(dp)

    #         Regular dp ,O(n^2)
    #         n = len(nums)
    #         dp = [0]*n
    #         #dp[i] is the length of longest increasing sequence that ends with nums[i]
    #         dp[0] = 1
    #         for i in range(1,n):
    #             temp = 0
    #             for j in range(i):
    #                 if nums[i]>nums[j]: temp = max(temp, dp[j]+1)
    #             if temp == 0: temp = 1
    #             dp[i] = temp  
    #         return dp[max]

```
## [673.最长递增子序列的个数](https://leetcode.cn/problems/number-of-longest-increasing-subsequence/description/)
```python

class Solution:
    def findNumberOfLIS(self, nums: List[int]) -> int:
        n, max_len, ans = len(nums),0 ,0
        dp = [0]*n
        cnt = [0]*n
        #dp[i] is the length of longest increasing sequence that ends with nums[i]
        dp[0] = 1
        cnt[0] =1 
        for i in range(n):
            cnt[i] = 1
            dp[i] = 1
            for j in range(i):
                if nums[i]>nums[j]: 
                    if dp[j]+1>dp[i]:
                        dp[i] = dp[j]+1
                        cnt[i] = cnt[j] 
                    elif dp[j]+1 == dp[i]:
                        cnt[i]+=cnt[j]
            # if dp[i]>max_len:
            #     max_len = dp[i]
            #     ans = cnt[i]
            # elif dp[i] == max_len:
            #     ans += cnt[i]
        print(cnt)
        return sum(cnt[i] for i in range(n) if dp[i] == max(dp))

```
## [354. 俄罗斯套娃信封问题](https://leetcode.cn/problems/russian-doll-envelopes/description/)

```python
from bisect import bisect_left

class Solution:
    def maxEnvelopes(self, envelopes: List[List[int]]) -> int:
        n = len(envelopes)
        dp = []
        envelopes.sort(key=lambda x: (x[0], -x[1]))
        target_list = [envelope[1] for envelope in envelopes]

        for num in target_list:
            if not dp or num>dp[-1]: dp.append(num)
            else:
                loc = bisect_left(dp, num)
                dp[loc] = num
        print(dp)
        return len(dp)


```
## [152. 乘积最大子数组](https://leetcode.cn/problems/maximum-product-subarray/description/)
```python

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        n = len(nums)
        dpmin = [0]*n
        dpmax = [0]*n
        #dpmin[i]代表以nums[i]结尾的子序列的最小乘积
        #dpmax[i]代表以nums[i]结尾的子序列的最大乘积
        dpmax[0] = nums[0]
        dpmin[0] = nums[0]
        for i in range(1,n):
            dpmax[i] = max(dpmax[i-1]*nums[i],dpmin[i-1]*nums[i],nums[i])
            dpmin[i] = min(dpmax[i-1]*nums[i],dpmin[i-1]*nums[i],nums[i])
        print(dpmax)
        print(dpmin)
        return max(dpmax)
```
## [53. 最大子数组和](https://leetcode.cn/problems/maximum-subarray/description/)
```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:

        n = len(nums)
        dp = [0]*n
        dp[0] = nums[0]
        for i in range(1,n):
            dp[i] = max(dp[i-1]+nums[i],nums[i])
                
        return max(dp)
```
## [918. 环形子数组的最大和](https://leetcode.cn/problems/maximum-sum-circular-subarray/description/)
```python
class Solution:
    def maxSubarraySumCircular(self, nums: List[int]) -> int:

        n = len(nums)
        dpmax = [0]*n
        dpmax[0] = nums[0]
        total_sum = nums[0]
        for i in range(1,n):
            total_sum += nums[i]
            dpmax[i] = max(dpmax[i-1]+nums[i],nums[i])

        dpmin = [0]*n
        dpmin[0] = nums[0]
        for i in range(1,n):
            dpmin[i] = min(dpmin[i-1]+nums[i],nums[i])
            
        if min(dpmin)==total_sum: return max(dpmax)
        else: return max(max(dpmax),total_sum-min(dpmin))
```


## [面试题 17.24. 最大子矩阵](https://leetcode.cn/problems/max-submatrix-lcci/description/)
```python
from typing import List

class Solution:
    def getMaxMatrix(self, matrix: List[List[int]]) -> List[int]:
        if not matrix or not matrix[0]:
            return []
        
        n, m = len(matrix), len(matrix[0])
        max_sum = float('-inf')
        result = [0, 0, 0, 0]  # [r1, c1, r2, c2]

        for r1 in range(n):
            # Initialize an array to hold the sum of columns
            col_sum = [0] * m
            for r2 in range(r1, n):
                for c in range(m):
                    col_sum[c] += matrix[r2][c]

                # Now find the max subarray in col_sum using Kadane's algorithm
                current_sum = 0
                start_col = 0
                for c in range(m):
                    if current_sum <= 0:
                        current_sum = col_sum[c]
                        start_col = c
                    else:
                        current_sum += col_sum[c]
                    
                    if current_sum > max_sum:
                        max_sum = current_sum
                        result = [r1, start_col, r2, c]
        
        return result
```
## [1388. 3n 块披萨](https://leetcode.cn/problems/pizza-with-3n-slices/description/)
环状数组中取不相邻的最大子集和

```python
class Solution:
    def maxSizeSlices(self, slices: List[int]) -> int:
        def maxSum(arr):
            n = len(arr)
            m = (n+1) // 3
            print(m,n)
            dp = [[-10**9] * (m + 1) for _ in range(n)]
            dp[0][0] ,dp[0][1] = 0, arr[0]
            dp[1][0] ,dp[1][1] = 0, max(arr[0],arr[1])
            for i in range(2, n):
                dp[i][0] = 0
                for j in range(1, m + 1):
                    dp[i][j] = max(dp[i - 1][j], dp[i - 2][j - 1] + arr[i])
            
            return dp[n-1][m]

        # Considering the array as a circle, solve the problem twice:
        # 1. Without the first element
        # 2. Without the last element
        # Return the maximum of the two results
        
        return max(maxSum(slices[1:]), maxSum(slices[0:-1]))
```
## [873. 最长的斐波那契子序列的长度](https://leetcode.cn/problems/length-of-longest-fibonacci-subsequence/description/)

```python

# dp[(j, i)] 表示以 arr[j] 和 arr[i] 结尾的最长斐波那契子序列的长度。

class Solution:
    def lenLongestFibSubseq(self, arr: List[int]) -> int:
        index_map = {num: i for i, num in enumerate(arr)}
        n = len(arr)
        dp = {}
        max_len = 0
        
        for i in range(n):
            for j in range(i):
                prev_num = arr[i] - arr[j]
                if prev_num < arr[j] and prev_num in index_map:
                    k = index_map[prev_num]
                    dp[(j, i)] = dp.get((k, j), 2) + 1
                    max_len = max(max_len, dp[(j, i)])
        
        return max_len if max_len >= 3 else 0
```
## [887. 鸡蛋掉落](https://leetcode.cn/problems/super-egg-drop/description/)
```latex
# Method1: 考虑如果已知有j个鸡蛋，可以操作i次，n最多是多少，然后找到最小使f(i,j)>=n的i即可
# 考虑2个鸡蛋的情况，如果可以操作5次： 那么第一次应该仍在5楼，因为即使碎掉，也只需1+4 = 5次来确定，这是一个f(4,1)的子问题；如果没碎，就转化为了f(4,2)的子问题；这样f(5,2) = 1 + f(4,1) + f(4,2)
```
```python

@cache
def dfs(i,j):
    if i==0 or j==0: return 0
    else:
        return dfs(i-1,j-1)+1+dfs(i-1,j)

class Solution:
    def superEggDrop(self, k: int, n: int) -> int:
        for i in count(1):
            if dfs(i,k)>=n:
                break
        return i
```
