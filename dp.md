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
Method1: 考虑如果已知有j个鸡蛋，可以操作i次，n最多是多少，然后找到最小使f(i,j)>=n的i即可

考虑2个鸡蛋的情况，如果可以操作5次： 那么第一次应该扔在5楼，因为即使碎掉，也只需1+4 = 5次来确定，这是一个f(4,1)的子问题；如果没碎，就转化为了f(4,2)的子问题；这样f(5,2) = 1 + f(4,1) + f(4,2)；最终答案是15

考虑3个鸡蛋，操作6次的情况： 第一次扔的楼层应该是16层，因为即使碎了，我们只需在1-15层中确认，而我们知道f(5,2)正好是15，恰好可以handle；如果没碎，那么就转化为了f(5,3);同样有f(6,3) = 1 + f(5,2) + f(5,3) 

有 f(i,j) = 1 + f(i-1,j-1) + f(i-1, j)  

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

对于j = 2的情况, 假设需要操作n次，那么第一次扔在n楼； 第二次扔在 n+ (n-1) 楼，因为即使碎了也只需要 2+ n-2 = n次； 第三次扔在n + (n-1) + (n-2) 楼

因此答案是最小的n，使得n*(n+1)/2> target

```python
class Solution:
    def twoEggDrop(self, n: int) -> int:
        return ceil((sqrt(n * 8 + 1) - 1) / 2)
```

Method2: 动态规划 + 二分查找
dp[i][j]为i个鸡蛋，j层楼的最少次数 有通项dp[i][j] = min(dp[i][j], max(dp[i][j-t], dp[i-1][t-1])+1), 但是复杂度为O(k*n^2)会超时
所以应该用记忆化的dp+binary search
```python
class Solution:
    def superEggDrop(self, k: int, n: int) -> int:
        #i eggs and j floor
        @cache
        def dfs(i,j):
            if i==1 : return j
            elif j==0: return 0
            else: 
                low,high = 1, j
                while low < high:
                    mid = (low+high)//2

                    left = dfs(i,j-mid)
                    right = dfs(i-1,mid-1)

                    if left == right:
                        low = high = mid
                    elif left<right:
                        high = mid
                    else: 
                        low = mid+1
                ans = 1 + max( dfs(i,j-low), dfs(i-1,low-1) )

            return ans
        return dfs(k,n)
```
[64.最小路径和](https://leetcode.cn/problems/minimum-path-sum/solutions/)
```python
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])
        dp = [[0 for _ in range(n)]for _ in range(m)]
        dp[0][0] = grid[0][0]
        temp = 0
        for s in range(m):
            temp += grid[s][0]
            dp[s][0] = temp 

        temp = 0
        for t in range(n):
            temp += grid[0][t]
            dp[0][t] = temp   

        for i in range(1,m):
            for j in range(1,n):
                dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j] 

        return dp[-1][-1]
```
[55.跳跃游戏](https://leetcode.cn/problems/jump-game/description/)
```python
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        n = len(nums)
        if n==1: return True
        temp = 0
        for i in range(n-1):
            if temp< i: return False
            temp = max(temp,i+nums[i])
            if temp >= n-1: return True
        return False
```
[45.跳跃游戏II](https://leetcode.cn/problems/jump-game-ii/)

每次找到可到达的最远位置，就可以在线性时间内得到最少的跳跃次数。
```python
class Solution:
    def jump(self, nums: List[int]) -> int:

        n = len(nums)
        if n==1 : return 0
        counter = 0
        temp = 0
        i = 0
        while i <= n-2:
            print(i)
            counter +=1
            temp = i + nums[i]
            if temp>=n-1: return counter
            if nums[i] ==1 : i = i+1
            else:
                temp_idx = i+1
                temp_val = nums[i+1]+i
                for j in range(i+1, temp+1):
                    if nums[j]+j>= temp_val: 
                        temp_idx = j
                        temp_val = nums[j]+j
                i = temp_idx
```
[131.分割回文串](https://leetcode.cn/problems/palindrome-partitioning/)
```python
#dp[i][j] 表示s[i:j]是否为palidrome
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        n = len(s)
        dp = [[True]*(n) for _ in range(n)]

        for i in range(n-1,-1,-1):
            for j in range(i+1,n):
                dp[i][j] = dp[i+1][j-1] and s[i]==s[j]
        
        ret = list()
        ans = list()

        def dfs(i):
            if i==n:
                ret.append(ans[:])
                return 
            for j in range(i,n):
                if dp[i][j]:
                    ans.append(s[i:j+1])
                    dfs(j + 1)
                    ans.pop()
        dfs(0)
        return ret

```

[132.分割回文串2](https://leetcode.cn/problems/palindrome-partitioning-ii/description/)
```python
#f[i]= 0≤j<imin{f[j]}+1,其中 s[j+1..i] 是一个回文串

class Solution:
    def minCut(self, s: str) -> int:
        n = len(s)
        dp1 = [[True]* n for _ in range(n)]
        
        for i in range(n-1, -1, -1):
            for j in range(i+1 , n):
                dp1[i][j] = dp1[i+1][j-1] and s[i]==s[j]

        dp2 = [10000]* n
        for i in range(n):
            if dp1[0][i]:
                dp2[i] = 0
            else:
                for j in range(i):
                    if dp1[j+1][i]:
                        dp2[i] = min(dp2[i],dp2[j]+1)
        
        return dp2[n-1]
```

[516.最长回文子序列](https://leetcode.cn/problems/longest-palindromic-subsequence/description/?envType=study-plan-v2&envId=dynamic-programming)
```python

class Solution:
    def longestPalindromeSubseq(self, s: str) -> int:
        n = len(s)
        # dp[i][j]: longest palindrome from s[i] to s[j]
        # if s[i] == s[j]: dp[i][j] = dp[i+1][j-1]+2
        # Else: dp[i][j] = max(dp[i+1][i+j], dp[i][i+j-1])

        
        dp = [[0]*n for _ in range(n)]

        for i in range(n):
            for j in range(n):
                if i==j: dp[i][j] =1 

        for j in range(1,n):
            for i in range(n-j-1,-1,-1):
                if s[i] == s[i+j]: dp[i][i+j] = dp[i+1][i+j-1]+2
                else: dp[i][i+j] = max(dp[i+1][i+j], dp[i][i+j-1])

        return dp[0][n-1]

```

[戳气球](https://leetcode.cn/problems/burst-balloons/description/)
```python
class Solution:
    #    两端插入1方便操作
    #    dp[i][j]表示戳爆i和j之间（开区间）所有气球能得到的金币，k表示(i,j)中间最后戳爆的那一个
    def maxCoins(self, nums: List[int]) -> int:
        nums.insert(0,1)
        nums.insert(len(nums),1)
        n = len(nums)
        dp = [[0]*n for _ in range(n)]

        for window in range(2, n):
            for i in range(0,n-window):
                for k in range(i+1,i+window):
                    dp[i][i+window] = max(dp[i][i+window], dp[i][k]+dp[k][i+window]+ nums[i]*nums[k]*nums[i+window])
        return dp[0][n-1]
```


[最长公共子序列](https://leetcode.cn/problems/longest-common-subsequence/)

```python
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:

        # dp[i][j] : text1[0:i] ; text2[0:j]的最长公共子序列
        # dp[i][j] = dp[i-1][j-1] + 1 if text1[i-1]==text2[j-1] else max(dp[i-1][j], dp[i][j-1])
        <img width="538" alt="Screenshot 2024-12-05 at 11 39 01" src="https://github.com/user-attachments/assets/343c5a9c-2904-4247-8e3a-4bdbdba4729e">

        m,n = len(text1), len(text2)
        dp = [[0]*(n+1) for _ in range(m+1)]
        
        for j in range(1,n+1):
            for i in range(1,m+1):
                if text1[i-1] == text2[j-1]:
                    dp[i][j] = dp[i-1][j-1]+1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        return dp[m][n]
```
[最长重复字串](https://leetcode.cn/problems/longest-repeating-substring/description/)
dp[i][j]是两个分别以i和j结尾的相同子串的最大长度，其中i永远小于j。所有状态的值均初始化为0，状态转移时，如果s[i]和s[j]不同就不必管，因为以i结尾和以j结尾不会是相同子串，如果s[i]和s[j]相同，那么dp[i][j]就等于dp[i-1][j-1]+1

```python
class Solution:
    def longestRepeatingSubstring(self, s: str) -> int:
        n = len(s)
        dp = [[0]*n for _ in range(n)]
        ans = 0
        for i in range(n):
            for j in range(i+1,n):
                if s[i] == s[j]:
                    dp[i][j] = dp[i-1][j-1] + 1 
                    ans = max(ans, dp[i][j])
        return ans
```
[最长重复字数组](https://leetcode.cn/problems/maximum-length-of-repeated-subarray/)
```python
class Solution:
    def findLength(self, nums1: List[int], nums2: List[int]) -> int:
        m = len(nums1)
        n = len(nums2)
        ans = 0 
        # dp[i][j] is nums1[0:i] , nums2[0:j]里最大的重复字数组
        dp = [[0]* (n+1) for _ in range(m+1)]
        for i in range(1,m+1):
            for j in range(1,n+1):
                if nums1[i-1] == nums2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                    ans = max(ans, dp[i][j])
                    
        return ans

```
[174. 地下城游戏](https://leetcode.cn/problems/dungeon-game/description/)
从右下向左上遍历，因为如果从左上向右下，加的血会影响后续决策。这样的动态规划是不满足「无后效性」的。
dp[i][j]=max(min(dp[i+1][j],dp[i][j+1])−dungeon(i,j),1)

```python

class Solution:
    def calculateMinimumHP(self, dungeon: List[List[int]]) -> int:
        m,n = len(dungeon),len(dungeon[0])

        dp = [[float('inf')] * (n + 1) for _ in range(m + 1)]
        dp[m][n - 1] = dp[m - 1][n] = 1  # 终点右边和下边的假设值

        for i in range(m-1,-1,-1):
            for j in range(n-1,-1,-1):
                temp = min(dp[i+1][j], dp[i][j+1]) - dungeon[i][j]
                dp[i][j] = max(temp, 1)
        
        return dp[0][0]

```
【115.不同的子序列](https://leetcode.cn/problems/distinct-subsequences/description/)
```python
class Solution:
    def numDistinct(self, s: str, t: str) -> int:
        #dp[i][j] 表示在 s[i:]的子序列中t[j:]出现的个数
        #倒着递推
        #如果s[i] = t[j], 可以选择与之匹配，那么就是dp[i+1][j+1]； 也可以不匹配，那么就是dp[i+1][j]
        m,n = len(s), len(t)
        if m<n : return 0

        dp = [[0]* (n+1) for _ in range(m+1)]
        for j in range(n-1,-1,-1):
            for i in range(m-n+j,-1,-1):
                if s[i] == t[j]: 
                    dp[i][j] = dp[i+1][j+1] + dp[i+1][j]
                    if j==n-1: dp[i][j] = dp[i+1][j] + 1
                else: dp[i][j] = dp[i+1][j]
        return dp[0][0]


```
[72.编辑距离](https://leetcode.cn/problems/edit-distance/)

dp[i][j]: number to transform word1[:i] to word2[:j]
if word1[i-1] = word2[j-1]: dp[i][j] = dp[i-1][j-1]
else: 三种操作的最小值。 插入: dp[i][j-1]  删除：dp[i-1][j] 替换：dp[i-1][j-1]
```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        m = len(word1)
        n = len(word2)
        dp = [[0]*(n+1) for _ in range(m+1)]
        for i in range(m+1):
            dp[i][0] = i
        for j in range(n+1):
            dp[0][j] = j
        for i in range(1,m+1):
            for j in range(1,n+1):
                if  word1[i-1] == word2[j-1]: dp[i][j] = dp[i-1][j-1]
                else: 
                    dp[i][j] = 1+ min(dp[i-1][j-1], dp[i][j-1], dp[i-1][j])
        
        return dp[m][n]

```

[712. 两个字符串的最小ASCII删除和](https://leetcode.cn/problems/minimum-ascii-delete-sum-for-two-strings/description/)
```python
class Solution:
    def minimumDeleteSum(self, s1: str, s2: str) -> int:
        m = len(s1)
        n = len(s2)
        dp = [[0]*(n+1) for _ in range(m+1)]
        for s in range(1,m+1):
            dp[s][0] = dp[s-1][0] + ord(s1[s-1])
        for t in range(1,n+1):
            dp[0][t] = dp[0][t-1] + ord(s2[t-1]) 
        for i in range(1,m+1):
            for j in range(1,n+1):
                if s1[i-1] == s2[j-1]: 
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j]+ord(s1[i - 1]), dp[i][j-1] + ord(s2[j-1]))
        
        return dp[m][n]
```
[91. 解码方法](https://leetcode.cn/problems/decode-ways/description/)
```python
class Solution:
    def numDecodings(self, s: str) -> int:
        n = len(s)
        f = [1] + [0] * n
        for i in range(1, n + 1):
            if s[i - 1] != '0':
                f[i] += f[i - 1]
            if i > 1 and s[i - 2] != '0' and int(s[i-2:i]) <= 26:
                f[i] += f[i - 2]
        return f[n]
```
[买卖股票的最佳时机 II]
```python
# dp[i][j]， i是天数，j是状态
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        length = len(prices)
        #dp[i][0]第i天持有股票
        #dp[i][1]第i天不持有股票 
        dp = [[0] * 2 for _ in range(length)]
        dp[0][0] = -prices[0]
        dp[0][1] = 0
        for i in range(1, length):
            dp[i][0] = max(dp[i-1][0], dp[i-1][1] - prices[i]) #注意这里是和121. 买卖股票的最佳时机唯一不同的地方
            dp[i][1] = max(dp[i-1][1], dp[i-1][0] + prices[i])
        return dp[-1][1]
```

[买卖股票的最佳时机 IV](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-iv/)

顺便输出number of trades used.
```python
class Solution:
    def maxProfit(self, k: int, prices: List[int]) -> int:
        #2k states
        n = len(prices)
        dp = [[0]*2*k for _ in range(n)]
        # 状态2m, 2m+1, m = [0,k-1]
        # 2m: 完成m次交易的基础上，再进行一次买操作
        # 2m+1: 完成m+1次操作
 
        for i in range(2*k):
            if i%2 == 0:
                dp[0][i] = -prices[0]

        for i in range(1,n):
            for j in range(2*k):
                if j%2 == 0:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1]- prices[i])
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1]+ prices[i])
        maxProfit = max(dp[-1])
        total_trades = 0
        for i in range(2*k):
            if dp[-1][i] == maxProfit:
                total_trades = (i+1)//2
                break
        print(maxProfit, total_trades)
        print(dp[-1])
        return max(dp[-1])
```
