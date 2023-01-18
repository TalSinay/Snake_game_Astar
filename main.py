import numpy as np
import random
import time
import heapq


def snake_move(snake, move_array, move=""):
    """
    if move=="LEFT":
      move_array = np.array([0,-1])
    elif move=="RIGHT":
      move_array = np.array([0,+1])
    elif move=="UP":
      move_array = np.array([-1,0])
    elif move=="DOWN":
      move_array = np.array([+1,0])
    """

    snake_head = np.array(snake[-1])
    snake_head += move_array
    snake_head = snake_head.tolist()
    snake.append(snake_head)
    snake.pop(0)

    return snake


def map_reload(snake, map_size):
    map = np.zeros((map_size, map_size), dtype=str)
    map[map == ""] = " "
    for i in snake:
        map[i[0]][i[1]] = "x"
    snake_head = snake[-1]
    map[snake_head[0]][snake_head[1]] = "X"

    return map


def make_apple(map):
    apple_where = np.where(map == " ")
    i = random.randint(0, apple_where[0].size - 1)
    apple = [apple_where[0][i], apple_where[1][i]]

    return apple


def check_snake_head(snake, apple, map):
    snake_head = snake[-1]
    if snake_head == apple:
        apple = make_apple(map)
        snake.insert(0, snake[0])
        return apple, snake
    else:
        return apple, snake


def gameOver(snake, map):
    snake_head = snake[-1]
    edge = map[0].size
    if snake.count(snake[-1]) == 1 and 0 <= snake_head[0] <= edge and 0 <= snake_head[1] <= edge:
        return True
    else:
        print("GAME OVER")
        return False


class PriorityQueue:
    def __init__(self):
        self.elements: list[tuple[float, T]] = []

    def empty(self) -> bool:
        return not self.elements

    def put(self, item, priority: float):
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        return heapq.heappop(self.elements)[1]


def neighbors(point, obstacle, map_size):
    x = point[0]
    y = point[1]
    neighbors = [[x + 1, y], [x - 1, y], [x, y + 1],
                 [x, y - 1]]  # +[[x+1,y-1],[x-1,y-1],[x+1,y+1],[x+1,y-1]] #4向移动或8向移动
    ans = []
    for i in range(len(neighbors)):
        if neighbors[i] not in obstacle and 0 <= neighbors[i][0] <= map_size - 1 and 0 <= neighbors[i][
            1] <= map_size - 1:
            ans.append(neighbors[i])
    return ans


def heuristic(goal, next):
    return abs(goal[0] - next[0]) + abs(goal[1] - next[1])


def turn_cost(father, current, next, start):
    if current != start:
        father_array = np.array(father)
        current_array = np.array(current)
        direction = current_array - father_array + current_array
        direction = direction.tolist()
        if next == direction:
            return 0
        else:
            return 0.4
    else:
        return 0


def a_star_search(obstacle, start, goal, map_size):
    frontier = PriorityQueue()
    frontier.put(start, 0)
    father = {}
    cost = {}
    father[str(start)] = None
    cost[str(start)] = 0

    while not frontier.empty():
        current = frontier.get()

        if current == goal:
            break

        for next in neighbors(current, obstacle, map_size):
            new_cost = cost[str(current)] + heuristic(current, next)
            if str(next) not in cost or new_cost < cost[str(next)]:
                cost[str(next)] = new_cost
                priority = new_cost + heuristic(goal,next)  # + turn_cost(father[str(current)],current,next,start) #+ (next[0]-current[0])*0.1 #代价函数
                frontier.put(next, priority)
                father[str(next)] = current

    return father, cost


def print_the_path(father, goal, start):
    path = []

    while goal != start:
        path.append(goal)
        if str(goal) in father:
            goal = father[str(goal)]
        else:
            return path

    path.append(start)
    return path
for i in range(100):
    map_size=18
    map = np.zeros((map_size,map_size),dtype=str)
    map[map==""]=" "

    snake=[[9,9],[9,10],[9,11]]

    for i in snake:
        map[i[0]][i[1]]="X"

    apple = make_apple(map)
    print(map,end="",flush=True)
    game = gameOver(snake,map)

    while game==True:
        father,cost = a_star_search(snake,snake[-1],apple,map_size)
        path=print_the_path(father,apple,snake[-1])
        if len(path)<2:
            map = map_reload(snake,map_size)
            map[apple[0]][apple[1]] = "O"
            print(map,flush=True)
            print("CANNOT FIND PATH,GAME OVER")
            break
        try:
            move_array=np.array(path[-2])-np.array(snake[-1])
        except IndexError:
            print("Game Over!")

        snake = snake_move(snake,move_array=move_array)
        try:
            map = map_reload(snake,map_size)
        except: IndexError("GAME OVER 2!")
        map[apple[0]][apple[1]] = "O"
        apple,snake = check_snake_head(snake,apple,map)

        print(map,flush=True)
        print("score:",len(snake)-3)
        game = gameOver(snake,map)
        time.sleep(0.005)



