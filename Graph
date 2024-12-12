[547. 省份数量](https://leetcode.cn/problems/number-of-provinces/description/?envType=study-plan-v2&envId=graph-theory)
```python
# DFS
遍历所有城市，对于每个城市，如果该城市尚未被访问过，则从该城市开始深度优先搜索，通过矩阵 isConnected 得到与该城市直接相连的城市有哪些，然后对这些城市继续深度优先搜索，直到同一个连通分量的所有城市都被访问到，即可得到一个省份。遍历完全部城市以后，即可得到连通分量的总数，即省份的总数。

class Solution:
    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        def dfs(i:int):
            for j in range(cities):
                if isConnected[i][j] == 1 and j not in visited:
                    visited.add(j)
                    dfs(j)
        cities = len(isConnected)
        visited = set()
        provinces = 0

        for i in range(cities):
            if i not in visited:
                dfs(i)
                provinces+=1

        return provinces



```
