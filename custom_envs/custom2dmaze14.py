# -*- coding: utf-8 -*-
# ���C�u�����̃C���|�[�g
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
    
# gym.Env���p������EasyMaze�N���X
class EasyMaze(gym.Env):
    # ���̊��ł�render�̃��[�h�Ƃ���rgb_array�݂̂�p�ӂ��Ă��邱�Ƃ�錾���Ă���
    # Gym��Wrapper�Ȃǂ���Q�Ƃ����\��������
    metadata = {'render.modes': ['rgb_array']}

    m = 0.1 # ���H�̎���̊O�g�̕�
    c = 0.1   # �e�Z���̕�

    agent_color = "blue"      # �G�[�W�F���g�̐F
    maze_color = "black"      # ���H�̐F
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
        
        # �s����ԂƂ���0����3�܂ł�4��ނ̗��U�l��ΏۂƂ���
        # ���Ȃ݂ɁA0��"left"�A1��"top"�A2�́hright�h�A3��"down"�ɑΉ�������
        self.action_space = gym.spaces.Discrete(4)

        # ��Ԃ̓G�[�W�F���g�����݂���Z���̈ʒu(12���)
        self.observation_space = gym.spaces.Discrete(12)

        # ������V�̒l��0����1�̊ԂƂ���
        self.reward_range = (0, 1)

    
    def reset(self):
        # ���H�̃X�^�[�g�ʒu��"s0"�Ƃ���
        self.STATE = [Vars.START[0], Vars.START[1]]
        Vars.GRID[Vars.GOALP[1]][Vars.GOALP[0]] = Vars.GOAL
        # ������Ԃ̔ԍ����ϑ��Ƃ��ĕԂ�
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
        # ���݂̏�Ԃƍs�����玟�̏�ԂɑJ��
        # self.state = self.dynamics[self.state][action]
        
        reward = 0  # ������V��0
        done = False
        # self.ACT[1] = self.ACT[0]
        ret = self.collisionCheck(self.STATE, self.OLDSTATE, action)
        
        # self.ACT[0] = action
        # �S�[�����"s11"�ɑJ�ڂ��Ă�����I���������Ƃ�done�Ɋi�[����V1���i�[
        # ���̑��̏�ԂȂ�done=False, reward=0�Ƃ���
        if self.STATE == Vars.GOALP:
            done = True
            reward = 1
        elif ret < 0:
            reward = -0.1
        # else:
        #     done = False
        #     reward = 0
        
        
        # ����̗�ł�info�͎g�p���Ȃ�
        info = {}
        

        return self.STATE, reward, done, info

    
     
    # ���H��`�悷��֐�
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

#         # �G�[�W�F���g�̍s���Ȃǂ̃��W�b�N�����s
#         action = env.action_space.sample()  # �����œK�؂ȍs����I�����郍�W�b�N������
#         obs, reward, done, info = env.step(action)
#         print(obs)
        
#         env.render()
#         pygame.display.update()
        
#         # �Q�[�����I�������ꍇ�A���Z�b�g
#         if done:
#             # break
#             # pygame.quit()
#             env.reset()
#               # Deploying the agent and goal.

        
    





    
    
    