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
DSatur 算法

个性化歌单推荐系统
假设你是音乐服务的开发者，为了提高用户体验需要解决推荐歌单的同质化问题，保证推荐给用户的所有歌单不包含相同歌曲。
给定一个包含 N 个歌单和 M 条歌单重复记录，每个歌单用一个从 1 到 N 的整数编号，歌单重复记录包含两个歌单的 ID，表示两个歌单有相同的歌曲。
你的任务是对歌单进行合并，找出合并后的最小歌单数量，合并的歌单中不能有相同的歌曲。
解答要求
时间限制：C/C++ 1000ms，其他语言：2000ms
内存限制：C/C++ 256MB，其他语言：512MB
输入
第一行包含两个整数 N 和 M，分别表示歌单的数量和有相同歌曲的歌单记录数。
接下来 M 行，每行包含两个整数编号 X 和 Y，表示编号为 X 和 Y 的歌单有相同的歌曲。
输入不会出现相同歌单，例如不会出现 “1 1” 这种输入
输入不会出现多条重复的记录，例如 “1 2” 不会出现多次
最大歌单数量不超过 100
歌单有重复歌曲记录数 M 不会超过 1004
歌单 1 和 2 有相同歌曲，歌单 2 和 3 有相同歌曲，歌单 1 和 3 不一定包含相同歌曲
输出
输出一个整数，表示合并后的最小歌单数量
样例 1
输入：
5 6
1 2
1 3
1 4
2 3
2 5
4 5
输出：
3
解释：
输入有 5 个歌单，歌单编码从 1 到 5；有 6 条重复歌单记录，每一条记录包含了歌单的编码。
1 和 2 有相同歌曲；1 和 3 有相同歌曲；1 和 4 有相同歌曲；2 和 3 有相同歌曲；2 和 5 有相同歌曲；4 和 5 有相同歌曲。
输出合并后最小歌单数为 3，合并后的 3 个歌单内没有相同歌曲
1 和 5 一组；3 和 4 一组；2 一组（或者 1 和 5 一组；2 和 4 一组；3 一组），合并组合可能有多种，只需要输出合并后的最小数。
样例 2
输入：
4 3
1 2
1 3
1 4
输出：
2
解释：2/3/4 一组，没有相同歌曲；1 一组。

···python

# Implementation of the test function
# 在每一步选择“饱和度（Saturation）”最高的顶点进行着色 如果有多个顶点饱和度相同，则优先选择度数最大的顶点；如果仍然有并列的情况，可以按照顶点编号或其他策略进行打破。
def min_playlist_count(N, M, edges):
    from collections import defaultdict
    
    # Construct adjacency list
    graph = defaultdict(set)
    for x, y in edges:
        graph[x].add(y)
        graph[y].add(x)
    
    # Initialize
    color = [-1] * (N + 1)  # Color array, -1 means uncolored
    saturation = [0] * (N + 1)  # Saturation array
    degrees = [len(graph[i]) for i in range(N + 1)]  # Degree array
    uncolored_nodes = set(range(1, N + 1))  # Set of uncolored nodes
    
    while uncolored_nodes:
        # Find the node with maximum saturation and degree
        node = max(uncolored_nodes, key=lambda x: (saturation[x], degrees[x]))
        # Find the smallest available color for the current node
        used_colors = {color[neighbor] for neighbor in graph[node] if color[neighbor] != -1}
        for c in range(N):
            if c not in used_colors:
                color[node] = c
                break
        # Update the saturation of neighbors
        for neighbor in graph[node]:
            if neighbor in uncolored_nodes:
                saturation[neighbor] += 1
        # Remove the colored node
        uncolored_nodes.remove(node)
    
    # Return the minimum number of colors used
    return max(color) + 1

# Test cases
test_cases = [
    (5, 6, [(1, 2), (1, 3), (1, 4), (2, 3), (2, 5), (4, 5)]),  # Expected output: 3
    (4, 3, [(1, 2), (1, 3), (1, 4)]),                        # Expected output: 2
]

# Execute test cases
results = [min_playlist_count(N, M, edges) for N, M, edges in test_cases]
results


```
