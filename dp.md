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
