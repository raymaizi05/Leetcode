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
