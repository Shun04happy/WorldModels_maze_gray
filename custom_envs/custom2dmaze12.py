# -*- coding: utf-8 -*-
# ���C�u�����̃C���|�[�g
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
    
# gym.Env���p������EasyMaze�N���X
class EasyMaze(gym.Env):
    # ���̊��ł�render�̃��[�h�Ƃ���rgb_array�݂̂�p�ӂ��Ă��邱�Ƃ�錾���Ă���
    # Gym��Wrapper�Ȃǂ���Q�Ƃ����\��������
    metadata = {'render.modes': ['rgb_array']}

    m = 0.1 # ���H�̎���̊O�g�̕�
    c = 0.1   # �e�Z���̕�

    agent_color = "blue"      # �G�[�W�F���g�̐F
    maze_color = "green"      # ���H�̐F

   
    def __init__(self,model=None):
        super(EasyMaze, self).__init__()

        self.fig = None
        self.ax = None
        
        
        self.STATE = [Vars.START[0], Vars.START[1]]       # (x, y)
        self.OLDSTATE = [Vars.START[0], Vars.START[1]]    # Old coordinates
        self.ACT = [0, 0, 0]
        
        # �s����ԂƂ���0����3�܂ł�4��ނ̗��U�l��ΏۂƂ���
        # ���Ȃ݂ɁA0��"left"�A1��"top"�A2�́hright�h�A3��"down"�ɑΉ�������
        self.action_space = gym.spaces.Discrete(4)

        # ��Ԃ̓G�[�W�F���g�����݂���Z���̈ʒu(12���)
        self.observation_space = gym.spaces.Discrete(12)

        # ������V�̒l��0����1�̊ԂƂ���
        self.reward_range = (0, 1)

    def reset(self):
        # ���H�̃X�^�[�g�ʒu��"s0"�Ƃ���
        self.STATE = Vars.START
        # ������Ԃ̔ԍ����ϑ��Ƃ��ĕԂ�
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
        # ���݂̏�Ԃƍs�����玟�̏�ԂɑJ��
        # self.state = self.dynamics[self.state][action]
        
        self.collisionCheck(self.STATE, self.OLDSTATE, action)
        # �S�[�����"s11"�ɑJ�ڂ��Ă�����I���������Ƃ�done�Ɋi�[����V1���i�[
        # ���̑��̏�ԂȂ�done=False, reward=0�Ƃ���
        if self.STATE == Vars.GOALP:
            done = True
            reward = 1
        else:
            done = False
            reward = 0
        
        
        # ����̗�ł�info�͎g�p���Ȃ�
        info = {}

        return self.STATE, reward, done, info

    # �`��֘A�̏��������{
    def render(self, mode='rgb_array'):
        # matplotlib��p���Ė��H���쐬
        self.make_maze()
        # ���݈ʒu�ɃG�[�W�F���g��z�u
        # self.plot_agent(self.state)
        # matplotlib�ō쐬�����}��z���RGB�z��ɕϊ�
        rgb_array = self.fig2array()[:, :, :3]
        # RGB�z������^�[��
        return rgb_array 
    
     # �`��֘A�̏��������{
    def render_agent(self, mode='rgb_array'):
        self.plot_agent(self.STATE)
        # matplotlib�ō쐬�����}��z���RGB�z��ɕϊ�
        rgb_array = self.fig2array()[:, :, :3]
        # RGB�z������^�[��
        return rgb_array 

    # ���H��`�悷��֐�
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

                
                
    # �G�[�W�F���g��`��
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

                
    # �O�̃G�[�W�F���g�̃v���b�g���N���A
    def clear_agent(self):
        if hasattr(self, 'agent_marker') and self.agent_marker:
            self.agent_marker[0].remove()
        
    # matplotlib�̉摜�f�[�^��numpy�ɕϊ� 
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
#         # ��Ԃ�����
#         # env.render_agent()
#         plt.draw()
#         plt.pause(0.00001)  # ��Ԃ��\������鎞�ԁi�K�v�ɉ����Ē����j
    
#     print("end")
#     plt.pause(5)
    
    
    