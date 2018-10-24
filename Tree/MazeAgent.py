import collections
import queue

#Annan Miao
#Reference https://eddmann.com/posts/depth-first-search-and-breadth-first-search-in-python/
#Reference https://www.redblobgames.com/pathfinding/a-star/implementation.html

class MazeAgent(object):
    '''
    Agent that uses path planning algorithm to figure out path to take to reach goal
    Built for Malmo discrete environment and to use Malmo discrete movements
    '''

    def __init__(self, grid, method):
        '''
        Arguments
            grid -- (optional) a 2D list that represents the map
        '''
        self.__frontier_set = None
        self.__explored_set = None
        self.__goal_state = 3
        self.__grid = grid

        self.pathmethod = method

    def get_eset(self):
        return self.__explored_set

    def get_fset(self):
        a = queue.PriorityQueue()
        if type(self.__frontier_set) == type(a):
            # In A* search the frontier set is a priority queue, we need to transfer it into a queue
            return self.__frontier_set.queue
        return self.__frontier_set

    def get_goal(self):
        return self.__goal_state

    def set_grid(self, grid):
        self.__grid = grid

    def getStartpoint(self):
        # return the position (row, column) of the start point 2
        start = 2
        grid = self.__grid
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == start:
                    return i, j

    def getGoalpoint(self):
        # return the position (row, column) of the goal point 3
        goal = 3
        grid = self.__grid
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == goal:
                    return i, j

    def __plan_path_breadth(self):
        '''Breadth-First tree search'''
        grid = self.__grid
        start = self.getStartpoint()
        goal = self.get_goal()
        wall = 0
        height = len(grid)
        width = len(grid[0])

        self.__frontier_set = collections.deque([[start]])  # Set the frontier set to be a queue
        self.__explored_set = set([start])
        while self.__frontier_set:
            path = self.__frontier_set.popleft()
            x, y = path[-1]
            if grid[x][y] == goal:
                command = self.getCommand(path)
                return command[::-1]  # # Reverse the command because MazeSim uses last command first
            for x2, y2 in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if 0 <= x2 < height and 0 <= y2 < width and grid[x2][y2] != wall and (
                x2, y2) not in self.__explored_set:
                    self.__frontier_set.append(path + [(x2, y2)])
                    self.__explored_set.add((x2, y2))


    def __plan_path_astar(self):
        '''A* tree search'''
        goal = 3
        start = self.getStartpoint()
        grid = self.__grid
        wall = 0
        height = len(grid)
        width = len(grid[0])
        heuristic = self.getHeuristics()
        
        self.__frontier_set = queue.PriorityQueue()  # Set the frontier set to be a priority queue
        self.__frontier_set.put([start], heuristic[start[0]][start[1]])
        self.__explored_set = set([start])
        while self.__frontier_set:
            path = self.__frontier_set.get()  # the priority queue would output the element with highest priority, in this case the min value of distance from goal
            x, y = path[-1]
            if grid[x][y] == goal:
                command = self.getCommand(path)
                return command[::-1]  # # Reverse the command because MazeSim uses last command first
            for x2, y2 in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if 0 <= x2 < height and 0 <= y2 < width and grid[x2][y2] != wall and (
                x2, y2) not in self.__explored_set:
                    self.__frontier_set.put(path + [(x2, y2)], heuristic[x2][y2])
                    self.__explored_set.add((x2, y2))

    def getCommand(self, path):
        # Input the path (cosisting of locations in grid) and output the corresponding commands
        command = []
        for i in range(len(path) - 1):
            x, y = path[i]
            x2, y2 = path[i + 1]
            if x2 == x + 1:
                command.append("movenorth 1")
            elif x2 == x - 1:
                command.append("movesouth 1")
            elif y2 == y + 1:
                command.append("movewest 1")
            elif y2 == y - 1:
                command.append("moveeast 1")
        return command

    def getHeuristics(self):
        # Calculate the heuristics by the distance from the current cell to the goal
        grid = self.__grid
        height = len(grid)
        width = len(grid[0])
        
        x, y = self.getGoalpoint()
        heuristics = []
        for i in range(len(grid)):
            heuristics.append([])
            for j in range(len(grid[0])):
                # Since we use priorit queue in astar search, the h(x) is larger when it is closer to the goal
                heuristics[i].append(height + width - abs(i - x) - abs(j - y))
        return heuristics

    def get_path(self):
        '''should return list of strings where each string gives movement command
            (these should be in order)
            Example:
             ["movenorth 1", "movesouth 1", "moveeast 1", "movewest 1"]
             (these are also the only four commands that can be used, you
             cannot move diagonally)
             On a 2D grid (list), "move north" would move us
             from, say, [0][0] to [1][0]
        '''

        if self.pathmethod == "bf":
            return self.__plan_path_breadth()
        elif self.pathmethod == "astar":
            return self.__plan_path_astar()
