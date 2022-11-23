# Student agent: Add your own agent here
import copy
import math
import random
import time

from agents.agent import Agent
from store import register_agent
import sys


@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        self.MCtree = []
        self.autoplay = True

    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """
        allowed_time = time.time() + 1.85 # give the step function around 1.85 seconds to perform MCTS
        self.MCtree = []  # Monte Carlo Tree

        root = State(None, my_pos, adv_pos, chess_board)  # initialize the root of the tree with parent = None

        # simulation on the root
        score = self.simulate(root, max_step)

        root.num_of_runs += 1  # update the number of runs
        root.points += score  # update the score obtained

        root.children = self.get_all_children(root, max_step)   # store all children of the root

        for child in root.children:
            self.run_mcts_steps(child, max_step)

        while time.time() < allowed_time:
            # selection
            selected_state = self.get_highest_uct_state()
            if len(selected_state.children) == 0:   # if state not expanded yet
                selected_state.children = self.get_all_children(selected_state, max_step)  # get all next states

            if len(selected_state.children) == 0 or len(selected_state.children) == selected_state.index:
                # if there is no next state possible or if parent is already fully expanded
                # update its uct value to 0 so that it doesn't get chosen in the next iteration
                selected_state.uct = 0
                continue

            selected_child = selected_state.children[selected_state.index]  # choose the next child to be expanded
            selected_state.index += 1  # update the index value of the parent
            self.run_mcts_steps(selected_child, max_step)  # run the steps of Monte Carlo Tree Search

        # at the end of the two seconds, we need to make a choice about the next move
        # we pick the move that has the highest ratio (number of points / number of runs)
        max_ratio = 0
        chosen_move = root.children[0]
        for child in root.children:
            if (child.points / child.num_of_runs) > max_ratio:
                chosen_move = child
                max_ratio = child.points / child.num_of_runs
        return chosen_move.my_pos, chosen_move.dir

    def run_mcts_steps(self, selected_child, max_step):

        # expansion of the tree
        self.MCtree.append(selected_child)

        # simulation on the selected child
        score = self.simulate(selected_child, max_step)

        # backpropagation of the score up to the root
        temp_state = selected_child
        while temp_state is not None:
            temp_state.num_of_runs += 1  # update the number of runs
            temp_state.points += score   # update the score obtained
            temp_state = temp_state.parent

        # update all UCT values in the tree
        for state in self.MCtree:
            state.uct = self.compute_uct(state)

    def compute_uct(self, state):
        # use the UCT formula with c = sqrt(2)
        return (state.points / state.num_of_runs) + math.sqrt(2 * math.log(state.parent.num_of_runs) / state.num_of_runs)

    def get_highest_uct_state(self):  # choose the state having the highest UCT value
        max_uct = -1
        selected_state = None
        for state in self.MCtree:
            if state.uct > max_uct:
                selected_state = state
                max_uct = selected_state.uct
        return selected_state

    # Very important method that returns all possible moves from the initial state
    def get_all_children(self, init_state, num_steps):
        # setting tree parent
        parent = init_state
        tree = [[parent.my_pos]]
        r_max = len(parent.board)
        c_max = len(parent.board)
        r, c = parent.my_pos
        chessboard = parent.board
        all_states = []

        # iterate through 4 sides of new position for barrier check
        for index, b in enumerate(chessboard[r][c]):

            board = copy.deepcopy(chessboard)
            # If there is not a barrier, add one and consider this a new possible state
            if (not b):
                self.update_board(board, (r, c), index)
                new_state = State(parent, parent.my_pos, parent.adv_pos, board)
                new_state.dir = index
                all_states.append(new_state)

        visited = [(r, c)]
        step = 0

        # iterate by num of steps
        while step < num_steps:
            # new position that can be explored in the next iteration
            new_positions = []
            # iterate through each node at the current depth
            for (r, c) in (tree[step]):

                # check if moving UP is legal
                if (r - 1 > -1) and (not chessboard[r][c][0]) and ((r - 1, c) != parent.adv_pos) and (
                        (r - 1, c) not in visited):

                    new_positions.append((r - 1, c))
                    visited.append((r - 1, c))

                    # iterate through 4 sides of new position for barrier check
                    for index, b in enumerate(chessboard[r - 1][c]):

                        board = copy.deepcopy(chessboard)
                        # If there is not a barrier, add one and consider this a new possible state
                        if (not b):
                            self.update_board(board, (r - 1, c), index)
                            new_state = State(parent, (r - 1, c), parent.adv_pos, board)
                            new_state.dir = index
                            all_states.append(new_state)

                # check if moving RIGHT is legal
                if (c + 1 < c_max) and (not chessboard[r][c][1]) and ((r, c + 1) != parent.adv_pos) and (
                        (r, c + 1) not in visited):

                    new_positions.append((r, c + 1))
                    visited.append((r, c + 1))

                    # iterate through 4 sides of new position for barrier check
                    for index, b in enumerate(chessboard[r][c + 1]):

                        board = copy.deepcopy(chessboard)

                        # If there is not a barrier, add one and consider this a new possible state
                        if (not b):
                            self.update_board(board, (r, c + 1), index)
                            new_state = State(parent, (r, c + 1), parent.adv_pos, board)
                            new_state.dir = index
                            all_states.append(new_state)

                # check if moving DOWN is legal
                if (r + 1 < r_max) and (not chessboard[r][c][2]) and ((r + 1, c) != parent.adv_pos) and (
                        (r + 1, c) not in visited):

                    new_positions.append((r + 1, c))
                    visited.append((r + 1, c))

                    # iterate through 4 sides of new position for barrier check
                    for index, b in enumerate(chessboard[r + 1][c]):

                        board = copy.deepcopy(chessboard)

                        # If there is not a barrier, add one and consider this a new possible state
                        if (not b):
                            self.update_board(board, (r + 1, c), index)
                            new_state = State(parent, (r + 1, c), parent.adv_pos, board)
                            new_state.dir = index
                            all_states.append(new_state)

                # check if moving LEFT is legal
                if (c - 1 > -1) and (not chessboard[r][c][3]) and ((r, c - 1) != parent.adv_pos) and (
                        (r, c - 1) not in visited):

                    new_positions.append((r, c - 1))
                    visited.append((r, c - 1))

                    # iterate through 4 sides of new position for barrier check
                    for index, b in enumerate(chessboard[r][c - 1]):

                        board = copy.deepcopy(chessboard)

                        # If there is not a barrier, add one and consider this a new possible state
                        if (not b):
                            self.update_board(board, (r, c - 1), index)
                            new_state = State(parent, (r, c - 1), parent.adv_pos, board)
                            new_state.dir = index
                            all_states.append(new_state)

            # Add all children found to the tree of new positions
            tree.append(new_positions)
            step += 1

        return all_states

    # Performs a simulation of the game starting from a Node in the tree
    def simulate(self, state, max_step):
        current_state = copy.deepcopy(state)
        new_board = copy.deepcopy(state.board)
        current_state.board = new_board
        my_turn = True

        # while the game is not finished, keep on choosing random moves for us and the opponent
        while not self.is_game_finished(current_state):
            if my_turn:
                # if it is our turn, we make the random move
                position, direction = self.random_step(current_state.board, current_state.my_pos, current_state.adv_pos, max_step)
                current_state.board = self.update_board(current_state.board, position, direction)
                current_state.my_pos = position
            else:
                # if it is the opponent's turn, they make the random move
                position, direction = self.random_step(current_state.board, current_state.adv_pos, current_state.my_pos, max_step)
                current_state.board = self.update_board(current_state.board, position, direction)
                current_state.adv_pos = position
            my_turn = not my_turn   # change turns

        score = self.get_score(current_state)   # get the score of the finished game and return it
        return score

    # Based on a state of the board, compute the score obtained by our player
    def get_score(self, state):
        my_pos = state.my_pos
        adv_pos = state.adv_pos
        board = state.board

        # count the number of tiles for each player using a DFS algorithm
        my_num_tiles = self.depth_first_search(my_pos, board)
        adv_num_tiles = self.depth_first_search(adv_pos, board)

        # determine the score based on the winner (player that has more tiles in their region)
        # our score policy is +1 for a win, +0.5 for a tie and +0 for a loss
        if my_num_tiles > adv_num_tiles:
            return 1
        elif my_num_tiles == adv_num_tiles:
            return 0.5
        elif my_num_tiles < adv_num_tiles:
            return 0

    # DFS algorithm to see how many tiles a player can reach from their initial position
    def depth_first_search(self, pos, board):
        number_of_tiles_won = 0
        visited = [pos]
        stack = [pos]

        while stack:
            current_cell = stack.pop()
            visited.append(current_cell)
            number_of_tiles_won = number_of_tiles_won + 1

            # 4 possible moves
            up = (current_cell[0] - 1, current_cell[1])
            right = (current_cell[0], current_cell[1] + 1)
            down = (current_cell[0] + 1, current_cell[1])
            left = (current_cell[0], current_cell[1] - 1)

            if self.is_legal_move(up, "up", board) and (up not in visited):
                # if up is a legal move and not already visited, append it to the stack
                stack.append(up)
            if self.is_legal_move(right, "right", board) and (right not in visited):
                # if right is a legal move and not already visited, append it to the stack
                stack.append(right)
            if self.is_legal_move(down, "down", board) and (down not in visited):
                # if down is a legal move and not already visited, append it to the stack
                stack.append(down)
            if self.is_legal_move(left, "left", board) and (left not in visited):
                # if left is a legal move and not already visited, append it to the stack
                stack.append(left)
        return number_of_tiles_won

    # function that checks the state of the board and decides whether the simulated game is finished
    def is_game_finished(self, state):
        root = state.my_pos
        goal = state.adv_pos
        board = state.board
        visited = [root]
        stack = [root]

        # DFS algorithm to check if there is a path between the two players
        while stack:
            current_cell = stack.pop()
            if current_cell == goal:
                # if we were able to reach the goal, it means there was a path between the two players
                # Therefore, the game is not finished. Return false
                return False

            visited.append(current_cell)

            # 4 possibles moves
            up = (current_cell[0] - 1, current_cell[1])
            right = (current_cell[0], current_cell[1] + 1)
            down = (current_cell[0] + 1, current_cell[1])
            left = (current_cell[0], current_cell[1] - 1)

            if self.is_legal_move(up, "up", board) and (up not in visited):
                # if up is a legal move and not already visited, append it to the stack
                stack.append(up)
            if self.is_legal_move(right, "right", board) and (right not in visited):
                # if right is a legal move and not already visited, append it to the stack
                stack.append(right)
            if self.is_legal_move(down, "down", board) and (down not in visited):
                # if down is a legal move and not already visited, append it to the stack
                stack.append(down)
            if self.is_legal_move(left, "left", board) and (left not in visited):
                # if left is a legal move and not already visited, append it to the stack
                stack.append(left)

        # if we were able to empty the stack without finding the goal, it means the game is finished. Return true
        return True

    # Check if the applied move was actually legal to see if we should insert it in the stack for the DFS algorithm
    def is_legal_move(self, position, direction, board):
        size = len(board)
        x,y = position

        if x >= size or y >= size or x < 0 or y < 0:  # if the new position is out of the chessboard
            return False

        if direction == "up":
            if board[x][y][2] or board[x + 1][y][0]: # if barrier is blocking the move up
                return False

        elif direction == "right":
            if board[x][y][3] or board[x][y - 1][1]: # if barrier is blocking the move to the right
                return False

        elif direction == "down":
            if board[x][y][0] or board[x - 1][y][2]: # if barrier is blocking the move down
                return False

        elif direction == "left":
            if board[x][y][1] or board[x][y + 1][3]: # if barrier is blocking the move to the left
                return False

        # if all conditions are respected, it means that it was a valid move. Return true
        return True

    # Code taken directly from the file random_agent.py provided. Used to perform simulations as part of the MCTS steps
    def random_step(self, chess_board, my_pos, adv_pos, max_step):
        # Moves (Up, Right, Down, Left)
        ori_pos = copy.deepcopy(my_pos)
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        steps = random.randint(0, max_step + 1)

        # Random Walk
        for _ in range(steps):
            r, c = my_pos
            dir = random.randint(0, 3)
            m_r, m_c = moves[dir]
            my_pos = (r + m_r, c + m_c)

            # Special Case enclosed by Adversary
            k = 0
            while chess_board[r, c, dir] or my_pos == adv_pos:
                k += 1
                if k > 300:
                    break
                dir = random.randint(0, 3)
                m_r, m_c = moves[dir]
                my_pos = (r + m_r, c + m_c)

            if k > 300:
                my_pos = ori_pos
                break

        # Put Barrier
        dir = random.randint(0, 3)
        r, c = my_pos
        while chess_board[r, c, dir]:
            dir = random.randint(0, 3)

        return my_pos, dir

    # update the board with the added barrier
    def update_board(self, chessboard, pos, direction):
        r = pos[0]
        c = pos[1]

        # if we are setting barrier up
        if direction == 0:
            # set down barrier for the cell above the current one
            chessboard[r - 1][c][2] = True
            # set up barrier for current cell
            chessboard[r][c][direction] = True

        # if we are setting barrier right
        if direction == 1:
            # set left barrier for the cell that is to the right of current cell
            chessboard[r][c + 1][3] = True
            # set right barrier for current cell
            chessboard[r][c][direction] = True

        # if we are setting barrier down
        if direction == 2:
            # set up barrier for the cell that is under current cell
            chessboard[r + 1][c][0] = True
            # set under barrier for current cell
            chessboard[r][c][direction] = True

        # if we are setting barrier left
        if direction == 3:
            # set right barrier for the cell that is to the left of current cell
            chessboard[r][c - 1][1] = True
            # set left barrier for current cell
            chessboard[r][c][direction] = True

        return chessboard


class State:
    ''' Class to represent node object in the tree for MCTS'''
    def __init__(self, parent, my_pos, adv_pos, board):
        self.num_of_runs = 0  # number of runs performed
        self.points = 0       # score obtained
        self.parent = parent  # store the parent for backpropagation
        self.children = []    # store a list of its children (i.e. possible moves from the initial state)
        self.my_pos = my_pos  # Our player's position
        self.adv_pos = adv_pos   # Our adversary's position
        self.board = board    # the location of the barriers in a 3D array
        self.uct = 0          # The UCT score of the node
        self.index = 0        # The index of the next child to expand in self.children
        self.dir = -1         # The direction of the last barrier that we put