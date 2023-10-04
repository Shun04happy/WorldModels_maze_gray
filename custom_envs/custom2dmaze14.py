# -*- coding: utf-8 -*-
# ライブラリのインポート
import numpy as np
import matplotlib.pyplot as plt
import gym
import pygame
import threading


class Vars:
    START = [10,1 ] # (x, y)
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
                     [1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

    CS = 25
    SCR_X = GRID.shape[1] * CS  # column
    SCR_Y = GRID.shape[0] * CS  # row
    SCR_RECT = pygame.Rect(0, 0, SCR_X, SCR_Y)  # Rect(left,top,width,height)
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
    maze_color = "black"      # 迷路の色
    goal_color = "green"
    road_color = "white"
   
    def __init__(self,model=None):
        super(EasyMaze, self).__init__()

        self.fig = None
        self.ax = None
        
        
        self.STATE = [Vars.START[0], Vars.START[1]]       # (x, y)
        self.OLDSTATE = [Vars.START[0], Vars.START[1]]    # Old coordinates
        self.ACT = [0, 0, 0]
        
        self.screen = pygame.display.set_mode(Vars.SCR_RECT.size)
        
        # 行動空間として0から3までの4種類の離散値を対象とする
        # ちなみに、0は"left"、1は"top"、2は”right”、3は"down"に対応させた
        self.action_space = gym.spaces.Discrete(4)

        # 状態はエージェントが存在するセルの位置(12種類)
        self.observation_space = gym.spaces.Discrete(12)

        # 即時報酬の値は0から1の間とした
        self.reward_range = (0, 1)

    
    def reset(self):
        # 迷路のスタート位置は"s0"とする
        self.STATE = [Vars.START[0], Vars.START[1]]
        Vars.GRID[Vars.GOALP[1]][Vars.GOALP[0]] = Vars.GOAL
        # 初期状態の番号を観測として返す
        return self.STATE

    def move(self, state, oState, actNum):
        if actNum == 0:     # Go to up
            state[1] -= 1   # Move on coordinates
        elif actNum == 1:   # Go to right(up)
            state[0] += 1
        elif actNum == 2:   # Go to down
            state[1] += 1
        elif actNum == 3:   # Go to left(down)
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
        
        reward = 0  # 初期報酬は0
        done = False
        # self.ACT[1] = self.ACT[0]
        ret = self.collisionCheck(self.STATE, self.OLDSTATE, action)
        
        # self.ACT[0] = action
        # ゴール状態"s11"に遷移していたら終了したことをdoneに格納＆報酬1を格納
        # その他の状態ならdone=False, reward=0とする
        if self.STATE == Vars.GOALP:
            done = True
            reward = 1
        elif ret < 0:
            reward = -0.1
        # else:
        #     done = False
        #     reward = 0
        
        
        # 今回の例ではinfoは使用しない
        info = {}
        

        return self.STATE, reward, done, info

    
     
    # 迷路を描画する関数
    def render(self):
        
        for x in range(0, Vars.GRID.shape[1]):
            for y in range(0, Vars.GRID.shape[0]):
                tmpRect = pygame.Rect(x * Vars.CS, y * Vars.CS, Vars.CS, Vars.CS)
                if Vars.GRID[y][x] == Vars.WALL:
                    pygame.draw.rect(self.screen, (0, 0, 0), tmpRect)        # Painting with black
                elif Vars.GRID[y][x] == Vars.ROAD:
                    pygame.draw.rect(self.screen, (255, 255, 255), tmpRect)  # Painting with white
                    pygame.draw.rect(self.screen, (0, 0, 0), tmpRect, 1)     # and black lines

                    
                elif Vars.GRID[y][x] == Vars.GOAL:
                    pygame.draw.rect(self.screen, (25, 135, 22), tmpRect)
                    pygame.draw.rect(self.screen, (0, 0, 0), tmpRect, 1)
                elif Vars.GRID[y][x] == Vars.AGNT:
                    pygame.draw.circle(self.screen, (0, 0, 255), \
                                       (int((x * Vars.CS) + (Vars.CS / 2)), \
                                        int((y * Vars.CS) + (Vars.CS / 2))), int(Vars.CS / 3))
                    pygame.draw.rect(self.screen, (0, 0, 0), tmpRect, 1)
        
        
            
                elif Vars.GRID[x][y] == Vars.GOAL:
                    pygame.draw.rect(self.screen, (25, 135, 22), tmpRect)
                    pygame.draw.rect(self.screen, (0, 0, 0), tmpRect, 1)
        
                    
                
                
   
    
# if __name__ == '__main__':
    
    
#     running = True
#     clock = pygame.time.Clock()

#     pygame.init()
#     env = EasyMaze()
#     env.reset()
#     while running:
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 running = False

#         # エージェントの行動などのロジックを実行
#         action = env.action_space.sample()  # ここで適切な行動を選択するロジックを実装
#         obs, reward, done, info = env.step(action)
#         print(obs)
        
#         env.render()
#         pygame.display.update()
        
#         # ゲームが終了した場合、リセット
#         if done:
#             # break
#             # pygame.quit()
#             env.reset()
#               # Deploying the agent and goal.

        
    





    
    
    