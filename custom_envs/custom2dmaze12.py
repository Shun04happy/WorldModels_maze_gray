# -*- coding: utf-8 -*-
# ライブラリのインポート
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import gym
import threading


class Vars:
    START = [10, 1] # (x, y)
    GOALP = [1, 10]
    GRID = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                    [1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1],
                    [1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1],
                    [1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1],
                    [1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1],
                    [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
                    [1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

    CS = 25
                # SCR_X = GRID.shape[1] * CS  # column
                # SCR_Y = GRID.shape[0] * CS  # row
                # SCR_RECT = pygame.Rect(0, 0, SCR_X, SCR_Y)  # Rect(left,top,width,height)
    ROAD = 0  # Identification number of Road in the world
    WALL = 1  # Identification number of Wall or Obstacles in the world
    GOAL = 2  # Identification number of Goal in the world
    AGNT = 3  # Identification number of Agent in the world
    
# gym.Envを継承したEasyMazeクラス
class EasyMaze(gym.Env):
    # この環境ではrenderのモードとしてrgb_arrayのみを用意していることを宣言しておく
    # GymのWrapperなどから参照される可能性がある
    metadata = {'render.modes': ['rgb_array']}

    m = 0.1 # 迷路の周りの外枠の幅
    c = 0.1   # 各セルの幅

    agent_color = "blue"      # エージェントの色
    maze_color = "green"      # 迷路の色

   
    def __init__(self,model=None):
        super(EasyMaze, self).__init__()

        self.fig = None
        self.ax = None
        
        
        self.STATE = [Vars.START[0], Vars.START[1]]       # (x, y)
        self.OLDSTATE = [Vars.START[0], Vars.START[1]]    # Old coordinates
        self.ACT = [0, 0, 0]
        
        # 行動空間として0から3までの4種類の離散値を対象とする
        # ちなみに、0は"left"、1は"top"、2は”right”、3は"down"に対応させた
        self.action_space = gym.spaces.Discrete(4)

        # 状態はエージェントが存在するセルの位置(12種類)
        self.observation_space = gym.spaces.Discrete(12)

        # 即時報酬の値は0から1の間とした
        self.reward_range = (0, 1)

    def reset(self):
        # 迷路のスタート位置は"s0"とする
        self.STATE = Vars.START
        # 初期状態の番号を観測として返す
        return self.STATE

    def move(self, state, oState, actNum):
        if actNum == 0:     # Go to up
            state[1] -= 1   # Move on coordinates
        elif actNum == 1:   # Go to right
            state[0] += 1
        elif actNum == 2:   # Go to down
            state[1] += 1
        elif actNum == 3:   # Go to left
            state[0] -= 1
        # Update the grid attributes
        Vars.GRID[oState[1]][oState[0]] = 0
        Vars.GRID[state[1]][state[0]] = 3
    
     # Action function with collision check of wall and obstacles.
    def collisionCheck(self, state, oState, actNum):
        # Storing coordinates in oState.
        oState[0] = state[0]
        oState[1] = state[1]
        # Action select without collision.
        if actNum == 0:  # Go to up
            if Vars.GRID[(state[1] - 1)][state[0]] == Vars.ROAD or \
                    Vars.GRID[(state[1] - 1)][state[0]] == Vars.GOAL:
                self.move(state, oState, actNum)
                return 0
            else:  # Collision
                return -1
        elif actNum == 1:  # Go to right
            if Vars.GRID[state[1]][(state[0] + 1)] == Vars.ROAD or \
                    Vars.GRID[state[1]][(state[0] + 1)] == Vars.GOAL:
                self.move(state, oState, actNum)
                return 0
            else:  # Collision
                return -1
        elif actNum == 2:  # Go to down
            if Vars.GRID[(state[1] + 1)][state[0]] == Vars.ROAD or \
                    Vars.GRID[(state[1] + 1)][state[0]] == Vars.GOAL:
                self.move(state, oState, actNum)
                return 0
            else:  # Collision
                return -1
        elif actNum == 3:  # Go to left
            if Vars.GRID[state[1]][(state[0] - 1)] == Vars.ROAD or \
                    Vars.GRID[state[1]][(state[0] - 1)] == Vars.GOAL:
                self.move(state, oState, actNum)
                return 0
            else:  # Collision
                return -1
        # elif actNum == 4:  # Stay
        #     return 0
        else:  # Stay
            return 0
    
   

    def step(self, action):
        # 現在の状態と行動から次の状態に遷移
        # self.state = self.dynamics[self.state][action]
        
        self.collisionCheck(self.STATE, self.OLDSTATE, action)
        # ゴール状態"s11"に遷移していたら終了したことをdoneに格納＆報酬1を格納
        # その他の状態ならdone=False, reward=0とする
        if self.STATE == Vars.GOALP:
            done = True
            reward = 1
        else:
            done = False
            reward = 0
        
        
        # 今回の例ではinfoは使用しない
        info = {}

        return self.STATE, reward, done, info

    # 描画関連の処理を実施
    def render(self, mode='rgb_array'):
        # matplotlibを用いて迷路を作成
        self.make_maze()
        # 現在位置にエージェントを配置
        # self.plot_agent(self.state)
        # matplotlibで作成した図を配列にRGB配列に変換
        rgb_array = self.fig2array()[:, :, :3]
        # RGB配列をリターン
        return rgb_array 
    
     # 描画関連の処理を実施
    def render_agent(self, mode='rgb_array'):
        self.plot_agent(self.STATE)
        # matplotlibで作成した図を配列にRGB配列に変換
        rgb_array = self.fig2array()[:, :, :3]
        # RGB配列をリターン
        return rgb_array 

    # 迷路を描画する関数
    def make_maze(self):
        self.fig = plt.figure(figsize=(5, 5), dpi=200)
        self.ax = plt.axes()
        self.ax.axis("off")

        
        
        
        for x in range(0, Vars.GRID.shape[1]):
            for y in range(0, Vars.GRID.shape[0]):
                # tmpRect = pygame.Rect(x * Vars.CS, y * Vars.CS, Vars.CS, Vars.CS)
                if Vars.GRID[y][x] == Vars.WALL:
                    # pygame.draw.rect(screen, (0, 0, 0), tmpRect)        # Painting with black
                    r = patches.Rectangle(xy=(y, x), 
                                  width=1,
                                  height=1, 
                                  color=self.maze_color,
                                  fill=True)
                    self.ax.add_patch(r)
                elif Vars.GRID[y][x] == Vars.ROAD:
                    # pygame.draw.rect(screen, (255, 255, 255), tmpRect)  # Painting with white
                    # pygame.draw.rect(screen, (0, 0, 0), tmpRect, 1)     # and black lines
                    r1 = patches.Rectangle(xy=(y, x), 
                                  width=1,
                                  height=1, 
                                  color="white",
                                  fill=True)
                    self.ax.add_patch(r1)
                    
                    r2 = patches.Rectangle(xy=(y, x), 
                                  width=1,
                                  height=1, 
                                  color=self.maze_color,
                                  fill=False)
                    self.ax.add_patch(r2)
                    
                    # Arrow painting based on the value function with action values.
                    # actionValue = self.maxQvalue([x, y], Agent.agt1.Q, 5)
                    # if actionValue != 0:
                    #     if actionValue > 1:
                    #         actionValue = 1  # Fitting to limit
                    #     if actionValue > 0:
                    #         actionValue = actionValue * 255.0
                    #         color = (255, 255 - actionValue, 255 - actionValue)
                    #         pygame.draw.rect(screen, color, tmpRect)
                    #         pygame.draw.rect(screen, (0, 0, 0), tmpRect, 1)

                # elif Vars.GRID[y][x] == Vars.GOAL:
                #     pygame.draw.rect(screen, (25, 135, 22), tmpRect)
                #     pygame.draw.rect(screen, (0, 0, 0), tmpRect, 1)
                elif Vars.GRID[y][x] == Vars.AGNT:
                    # pygame.draw.circle(screen, (0, 0, 255), \
                    #                    (int((x * Vars.CS) + (Vars.CS / 2)), \
                    #                     int((y * Vars.CS) + (Vars.CS / 2))), int(Vars.CS / 3))
                    # pygame.draw.rect(screen, (0, 0, 0), tmpRect, 1)
                    self.agent_marker = self.ax.plot(y, x,
                              marker="o",
                              color=self.agent_color,
                              markersize=30)

                
                
    # エージェントを描画
    def plot_agent(self, state_name):
        self.clear_agent()
        # plt.pause(1)
        # state_index = self.maze_state_pos["text"].index(state_name)
        # agent_pos = self.maze_state_pos["xy"][state_index]
        # self.agent_marker = self.ax.plot([agent_pos[0]], 
        #                      [agent_pos[1]],
        #                       marker="o",
        #                       color=self.agent_color,
        #                       markersize=30)
        for x in range(0, Vars.GRID.shape[1]):
            for y in range(0, Vars.GRID.shape[0]):
                if Vars.GRID[y][x] == Vars.AGNT:
                    # pygame.draw.circle(screen, (0, 0, 255), \
                    #                    (int((x * Vars.CS) + (Vars.CS / 2)), \
                    #                     int((y * Vars.CS) + (Vars.CS / 2))), int(Vars.CS / 3))
                    # pygame.draw.rect(screen, (0, 0, 0), tmpRect, 1)
                    self.agent_marker = self.ax.plot(y+0.5, x+0.5,
                              marker="o",
                              color=self.agent_color,
                              markersize=10)

                
    # 前のエージェントのプロットをクリア
    def clear_agent(self):
        if hasattr(self, 'agent_marker') and self.agent_marker:
            self.agent_marker[0].remove()
        
    # matplotlibの画像データをnumpyに変換 
    def fig2array(self):
        self.fig.canvas.draw()
        w, h = self.fig.canvas.get_width_height()
        buf = np.fromstring(self.fig.canvas.tostring_argb(), dtype=np.uint8)
        buf.shape = (w, h, 4)
        buf = np.roll(buf, 3, axis=2)
        return buf
    
# if __name__ == '__main__':
    
#     env = EasyMaze()
    
#     obs = env.reset()
#     env.render()
#     # plt.pause(3)
#     # env.render_agent()
#     for _ in range(1000):
        
#         action = env.action_space.sample()
#         # print(action)
#         obs, re, done, info = env.step(action)
#         # env.render()
#         env.render_agent()
#         if done:
#             # env.reset()
#             plt.pause(5)
#             plt.close()
#         # 状態を可視化
#         # env.render_agent()
#         plt.draw()
#         plt.pause(0.00001)  # 状態が表示される時間（必要に応じて調整）
    
#     print("end")
#     plt.pause(5)
    
    
    