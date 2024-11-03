## [300. Longest Increasing Subsequence](https://leetcode.com/problems/longest-increasing-subsequence/)

```python
    def lengthOfLIS(self, nums: List[int]) -> int:
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
