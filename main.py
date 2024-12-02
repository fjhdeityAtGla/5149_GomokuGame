# use monte carlo tree search
# @author:Junhui Fu

import numpy as np
from collections import defaultdict
import random


class Gomoku:
    def __init__(self, board_size=15):
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=int)  # 0 空位，1 黑棋，2 白棋
        self.current_player = 1  # 黑棋先手

    def get_valid_actions(self):
        """返回所有空位作为合法动作。"""
        actions = [(i, j) for i in range(self.board_size) for j in range(self.board_size) if self.board[i, j] == 0]
        return actions

    def play_move(self, action):
        """执行给定动作。"""
        x, y = action
        self.board[x, y] = self.current_player
        self.current_player = 3 - self.current_player  # 切换玩家

    def undo_move(self, action):
        """撤销给定动作。"""
        x, y = action
        self.board[x, y] = 0
        self.current_player = 3 - self.current_player  # 切换回前一玩家

    def check_winner(self):
        """检查当前棋盘是否有胜者（五子连珠）。"""
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
        if not any(0 in row for row in self.board):  # 棋盘已满
            return 0  # 平局
        return -1  # 游戏未结束

    def display(self):
        """打印棋盘。"""
        for row in self.board:
            print(" ".join(str(x) if x > 0 else "." for x in row))
        print()


class AOAP_MCTS_Gomoku:
    def __init__(self, gomoku, n_rollouts=100, n0=10, epsilon=1e-5, prior_mean=0, prior_variance=10):
        self.gomoku = gomoku
        self.n_rollouts = n_rollouts
        self.n0 = n0
        self.epsilon = epsilon
        self.prior_mean = prior_mean
        self.prior_variance = prior_variance

        self.tree = defaultdict(lambda: {'visits': 0, 'value': self.prior_mean, 'variance': self.prior_variance})
        self.children = defaultdict(list)

    def expand_node(self, node):
        """扩展节点，添加所有可能动作。"""
        if node not in self.children:
            actions = self.gomoku.get_valid_actions()
            self.children[node] = actions
            for action in actions:
                child_node = f"{node}-{action}"
                self.tree[child_node]

    def aoap_select(self, node):
        """使用 AOAP 策略选择最佳动作。"""
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
        """随机模拟到游戏结束并返回奖励。"""
        while self.gomoku.check_winner() == -1:
            actions = self.gomoku.get_valid_actions()
            action = random.choice(actions)
            self.gomoku.play_move(action)

        winner = self.gomoku.check_winner()
        return 1 if winner == 2 else (-1 if winner == 1 else 0)

    def backpropagate(self, search_path, reward):
        """回溯更新路径中所有节点的统计值。"""
        for node in reversed(search_path):
            self.tree[node]['visits'] += 1
            old_mean = self.tree[node]['value']
            visits = self.tree[node]['visits']

            self.tree[node]['value'] += (reward - old_mean) / visits
            self.tree[node]['variance'] = ((visits - 1) * self.tree[node]['variance'] +
                                           (reward - old_mean) * (reward - self.tree[node]['value'])) / visits

    def tree_policy(self, node):
        """从根节点沿树策略选择路径。"""
        search_path = []
        while node in self.children and len(self.children[node]) > 0:
            search_path.append(node)
            action = self.aoap_select(node)
            self.gomoku.play_move(action)  # 修复 eval 问题
            node = f"{node}-{action}"

        return node, search_path

    def run(self):
        """运行 AOAP-MCTS 算法。"""
        root = 'root'
        self.expand_node(root)

        for _ in range(self.n_rollouts):
            self.gomoku = Gomoku(self.gomoku.board_size)  # 重置游戏
            leaf, search_path = self.tree_policy(root)
            self.expand_node(leaf)
            reward = self.default_policy()
            self.backpropagate(search_path, reward)

        best_action = max(self.children[root], key=lambda a: self.tree[f"{root}-{a}"]['value'])
        return best_action


def simulate_games(n_games=100):
    """模拟白棋与随机黑棋对战100次。"""
    white_wins = 0
    black_wins = 0
    draws = 0

    for game_idx in range(1, n_games + 1):
        print(f"游戏 {game_idx} 开始")
        gomoku = Gomoku(board_size=15)
        mcts = AOAP_MCTS_Gomoku(gomoku, n_rollouts=500)

        while gomoku.check_winner() == -1:
            # 黑棋随机下
            actions = gomoku.get_valid_actions()
            black_action = random.choice(actions)
            gomoku.play_move(black_action)
            print("黑棋落子:", black_action)
            gomoku.display()

            # 检查胜负
            if gomoku.check_winner() != -1:
                break

            # 白棋通过 MCTS 决策
            best_action = mcts.run()
            gomoku.play_move(best_action)
            print("白棋落子:", best_action)
            gomoku.display()

        # 显示本局结果
        winner = gomoku.check_winner()
        if winner == 1:
            print("本局结果: 黑棋胜利!")
            black_wins += 1
        elif winner == 2:
            print("本局结果: 白棋胜利!")
            white_wins += 1
        else:
            print("本局结果: 平局!")
            draws += 1

        print("-" * 30)

    # 显示最终统计
    print(f"白棋胜利: {white_wins}, 黑棋胜利: {black_wins}, 平局: {draws}")


if __name__ == "__main__":
    simulate_games()
