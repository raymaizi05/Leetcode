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
[841. 钥匙和房间](https://leetcode.cn/problems/keys-and-rooms/description/?envType=study-plan-v2&envId=graph-theory)
```python

def canVisitAllRooms(self, rooms: List[List[int]]) -> bool:
    #DFS
    visited = set()
    def dfs(room): 
        m = len(room)
        for i in range(m):
            temp = room[i]

            if temp not in visited: 
                visited.add(temp)
                dfs(rooms[temp])

    n = len(rooms)
    
    dfs(rooms[0])

    visited.add(0)
    print(visited)
    return len(visited)==n
    

def canVisitAllRooms(self, rooms: List[List[int]]) -> bool:
    #BFS
    n = len(rooms)
    num = 0
    vis = {0}
    que = collections.deque([0])

    while que:
        x = que.popleft()
        num += 1
        for it in rooms[x]:
            if it not in vis:
                vis.add(it)
                que.append(it)
    
    return num == n

```
[1129.颜色交替的最短数量](https://leetcode.cn/problems/shortest-path-with-alternating-colors/description/?envType=study-plan-v2&envId=graph-theory)
```python
class Solution:
    def shortestAlternatingPaths(self, n: int, redEdges: List[List[int]], blueEdges: List[List[int]]) -> List[int]:
        # 构建邻接表 g，其中 g[i] 存储所有从节点 i 出发的 (目标节点, 边颜色) 元组
        # 边颜色用0表示红色，1表示蓝色
        g = [[] for _ in range(n)]
        
        # 将红边加入邻接表
        for x, y in redEdges:
            g[x].append((y, 0))
        
        # 将蓝边加入邻接表
        for x, y in blueEdges:
            g[x].append((y, 1))
        
        # ans用于存储从0到各节点的最短交替路径长度，初始化为-1表示暂未得知
        ans = [-1] * n

        # vis用于记录已访问过的状态 (node, color)，
        # 这里的状态表示：到达 "node" 节点时使用了 "color" 颜色的边
        # 因为起点0没有上一条边，所以将起点0用两种状态初始化（假设起点可红可蓝）
        vis = {(0, 0), (0, 1)}

        # prev队列用于存储当前层待处理的节点状态
        # 初始化包含 (0,0) 和 (0,1)，即假设从0开始可以选择红边或蓝边作为第一步
        prev = [(0, 0), (0, 1)]

        # level表示BFS层数，BFS每一层对应从0出发路径长度递增1
        level = 0

        # BFS循环，当prev不为空，说明还有下一层的节点状态可以处理
        while prev:
            # 将当前层的节点状态集保存在temp中处理
            temp = prev
            # 重置prev，用于收集下一层的节点状态
            prev = []

            # 遍历当前层的所有状态
            for node, color in temp:
                # 如果该节点的最短距离还未确定（为-1），
                # 则当前的level就是从0到node的最短交替路径长度
                if ans[node] == -1:
                    ans[node] = level

                # 从node出发，尝试扩展下一层节点
                # g[node]中存储所有可到达的 (nd, cl) (目标节点nd, 边颜色cl)
                for nd, cl in g[node]:
                    # 条件1：该状态 (nd, cl) 未访问过
                    # 条件2：cl != color，保证颜色交替
                    if (nd, cl) not in vis and cl != color:
                        vis.add((nd, cl))   # 标记该状态已访问
                        prev.append((nd, cl))  # 加入下一层的队列

            # 当前层处理完毕，层数加1，代表路径长度增加
            level += 1

        # 返回结果数组ans，其中ans[i]即为到达节点i的最短交替路径长度
        # 若ans[i]仍为-1，说明无法以交替颜色路径从0到达i
        return ans



```
