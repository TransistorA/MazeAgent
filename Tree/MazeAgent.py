import collections


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

        self.__frontier_set = collections.deque([[start]])
        self.__explored_set = set([start])
        while self.__frontier_set:
            path = self.__frontier_set.popleft()
            x, y = path[-1]
            if grid[x][y] == goal:
                command = self.getCommand(path)
                return command
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

        self.__frontier_set = collections.deque([[start]])
        self.__explored_set = set([start])
        while self.__frontier_set:
            path = self.__frontier_set.popleft()
            x, y = path[-1]
            if grid[x][y] == goal:  # At the goal point
                command = self.getCommand(path)
                return command

            neibourset = []  # Clear the set after every move
            heuset = []
            for x2, y2 in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if 0 <= x2 < height and 0 <= y2 < width and grid[x2][y2] != wall and (
                x2, y2) not in self.__explored_set:
                    neibourset.append([x2, y2])
                    heuset.append(heuristic[x2][y2])  # a set of heuristics of all elements in the frontier set

            # Find the minimal heuristics
            minheu = heuset[0]
            minheuIndex = 0
            for i in range(len(heuset)):
                if heuset[i] < minheu:
                    minheuIndex = i
            x2, y2 = neibourset[minheuIndex]  # The next move with minimal heuristics value
            self.__frontier_set.append(path + [(x2, y2)])
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
        x, y = self.getGoalpoint()
        heuristics = []
        for i in range(len(grid)):
            heuristics.append([])
            for j in range(len(grid[0])):
                heuristics[i].append(abs(i - x) + abs(j - y))
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


a = MazeAgent(None, "bf")
a.set_grid([[3, 1, 2], [1, 1, 1], [1, 1, 1]])
print([3, 1, 2])
print([1, 1, 1])
print([1, 1, 1])
a.get_path()
