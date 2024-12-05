# use monte carlo tree search
# @author:Junhui Fu

import numpy as np
from collections import defaultdict
import random

from scipy.special import stirling2


class Gomoku:
    def __init__(self, board_size=15):
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=int)  # 0: empty, 1: black, 2: white
        self.current_player = 1  # Black starts first

    def get_valid_actions(self):
        """Return all empty positions as valid actions."""
        actions = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i, j] == 0:  # Select only empty positions
                    actions.append((i, j))
        return actions

    def play_move(self, action):
        """Execute the given action."""
        x, y = action
        self.board[x, y] = self.current_player
        self.current_player = 3 - self.current_player  # Switch player

    def check_winner(self):
        """Check if there is a winner (five in a row)."""
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for x in range(self.board_size):
            for y in range(self.board_size):
                if self.board[x, y] == 0:
                    continue
                player = self.board[x, y]
                for dx, dy in directions:
                    count = 1
                    for step in range(1, 5):
                        nx, ny = x + dx * step, y + dy * step
                        if 0 <= nx < self.board_size and 0 <= ny < self.board_size and self.board[nx, ny] == player:
                            count += 1
                        else:
                            break
                    if count >= 5:
                        return player
        if not any(0 in row for row in self.board):  # Board is full
            return 0  # Draw
        return -1  # Game not finished

    def display(self):
        """Print the board."""
        for row in self.board:
            print(" ".join(str(x) if x > 0 else "." for x in row))
        print()


class AOAP_MCTS_Gomoku:
    def __init__(self, gomoku, n_rollouts=100, epsilon=1e-5, prior_mean=0, prior_variance=10):
        self.gomoku = gomoku
        self.n_rollouts = n_rollouts
        self.epsilon = epsilon
        self.prior_mean = prior_mean
        self.prior_variance = prior_variance

        self.tree = defaultdict(lambda: {'visits': 0, 'value': self.prior_mean, 'variance': self.prior_variance})
        self.children = defaultdict(list)

    def expand_node(self, node):
        """Expand a node by adding all possible actions."""
        if node not in self.children:
            actions = self.gomoku.get_valid_actions()
            self.children[node] = actions
            for action in actions:
                child_node = f"{node}-{action}"
                self.tree[child_node]

    def aoap_select(self, node):
        """Use AOAP strategy to select the best action."""
        actions = self.children[node]
        max_vfa = -np.inf
        best_action = None

        for action in actions:
            child_node = f"{node}-{action}"
            info = self.tree[child_node]
            mean = info['value']
            variance = info['variance']
            visits = info['visits']

            vfa = mean + np.sqrt((variance + self.epsilon) / (visits + 1))
            if vfa > max_vfa:
                max_vfa = vfa
                best_action = action

        return best_action

    def default_policy(self):
        """Simulate to the end of the game randomly and return the reward."""
        simulated_board = Gomoku(self.gomoku.board_size)
        simulated_board.board = self.gomoku.board.copy()
        while simulated_board.check_winner() == -1:
            actions = simulated_board.get_valid_actions()
            action = random.choice(actions)
            simulated_board.play_move(action)

        winner = simulated_board.check_winner()
        return 1 if winner == 2 else (-1 if winner == 1 else 0)

    def backpropagate(self, search_path, reward):
        """Backpropagate the reward and update statistics for all nodes in the path."""
        for node in reversed(search_path):
            self.tree[node]['visits'] += 1
            old_mean = self.tree[node]['value']
            visits = self.tree[node]['visits']

            self.tree[node]['value'] += (reward - old_mean) / visits
            self.tree[node]['variance'] = ((visits - 1) * self.tree[node]['variance'] +
                                           (reward - old_mean) * (reward - self.tree[node]['value'])) / visits

    def tree_policy(self, node):
        """Select a path from the root using the tree policy."""
        search_path = []
        simulated_board = Gomoku(self.gomoku.board_size)
        simulated_board.board = self.gomoku.board.copy()
        while node in self.children and len(self.children[node]) > 0:
            search_path.append(node)
            action = self.aoap_select(node)
            simulated_board.play_move(action)
            node = f"{node}-{action}"

        return node, search_path

    def run(self):
        """Run the AOAP-MCTS algorithm."""
        root = 'root'
        self.expand_node(root)

        for _ in range(self.n_rollouts):
            leaf, search_path = self.tree_policy(root)
            self.expand_node(leaf)
            reward = self.default_policy()
            self.backpropagate(search_path, reward)

        best_action = max(self.children[root], key=lambda a: self.tree[f"{root}-{a}"]['value'])
        valid_actions = self.gomoku.get_valid_actions()
        if best_action not in valid_actions:
            print(f"Warning: MCTS selected an illegal action {best_action}, choosing a random valid action!")
            best_action = random.choice(valid_actions)  # Fallback to a random valid action

        return best_action


def simulate_games(n_games=100):
    """Simulate 100 games where white uses MCTS and black plays randomly."""
    white_wins = 0
    black_wins = 0
    draws = 0

    for game_idx in range(1, n_games + 1):
        print(f"Game {game_idx} starts")
        real_gomoku = Gomoku(board_size=15)

        while real_gomoku.check_winner() == -1:
            # Black moves randomly
            actions = real_gomoku.get_valid_actions()
            black_action = random.choice(actions)
            real_gomoku.play_move(black_action)
            print("Black plays:", black_action)

            # Check for game result
            if real_gomoku.check_winner() != -1:
                break
            mcts = AOAP_MCTS_Gomoku(real_gomoku, n_rollouts=500)
            # White moves using MCTS
            best_action = mcts.run()
            real_gomoku.play_move(best_action)
            print("White plays:", best_action)
            #real_gomoku.display()

        # Display the result of the game
        winner = real_gomoku.check_winner()
        if winner == 1:
            # print("本局结果: 黑棋胜利!")
            print("Result: Black wins!")
            black_wins += 1
        elif winner == 2:
            # print("本局结果: 白棋胜利!")
            print("Result: White wins!")
            white_wins += 1
        else:
            # print("本局结果: 平局!")
            print("Result: Draw!")
            draws += 1

        print("-" * 30)

    # Display final statistics
    # print(f"白棋胜利: {white_wins}, 黑棋胜利: {black_wins}, 平局: {draws}")
    print(f"White wins: {white_wins}, Black wins: {black_wins}, Draws: {draws}")


if __name__ == "__main__":
    simulate_games()

