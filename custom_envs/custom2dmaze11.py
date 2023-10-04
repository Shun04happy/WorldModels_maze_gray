# -*- coding: utf-8 -*-
# ���C�u�����̃C���|�[�g
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import gym
from gym import spaces

# gym.Env���p������EasyMaze�N���X
class EasyMaze(gym.Env):
    # ���̊��ł�render�̃��[�h�Ƃ���rgb_array�݂̂�p�ӂ��Ă��邱�Ƃ�錾���Ă���
    # Gym��Wrapper�Ȃǂ���Q�Ƃ����\��������
    metadata = {'render.modes': ['rgb_array']}

    m = 0.2 # ���H�̎���̊O�g�̕�
    c = 1   # �e�Z���̕�

    agent_color = "blue"      # �G�[�W�F���g�̐F
    maze_color = "green"      # ���H�̐F

    # ���H�̘g�̕`��֘A���
    maze_info_rec = {"xy":[(0, 0), (0, m+4*c), (m+4*c, 0), (0, 0),
                            (m, m+c), (m+c, m+3*c), (m+3*c, m+c)], 
                    "width":[m, 2*m+4*c, m, 2*m+4*c,
                                2*c, c, c], 
                    "height":[2*m+4*c, m, 2*m+4*c, m,
                                c, c, c]}

    # ���H���̓_���̕\���֘A���
    maze_info_line = {"s_xy":[(m, m+c), (m, m+2*c), (m, m+3*c),
                            (m+c, m), (m+2*c, m), (m+3*c, m)], 
                    "e_xy":[(m+4*c, m+c), (m+4*c, m+2*c), (m+4*c, m+3*c),
                            (m+c, m+4*c), (m+2*c, m+4*c), (m+3*c, m+4*c)]}

    # ��ԃe�L�X�g�̕\���ʒu���
    maze_state_pos = {"xy":[(m+0.5*c, m+3.5*c), (m+0.5*c, m+2.5*c), (m+1.5*c, m+2.5*c),
                            (m+2.5*c, m+2.5*c), (m+2.5*c, m+3.5*c), (m+3.5*c, m+3.5*c),
                            (m+3.5*c, m+2.5*c), (m+2.5*c, m+1.5*c), (m+2.5*c, m+0.5*c),
                            (m+3.5*c, m+0.5*c), (m+1.5*c, m+0.5*c), (m+0.5*c, m+0.5*c),], 
                        "text":["s0", "s1", "s2", "s3", "s4", "s5", "s6", 
                                "s7", "s8", "s9", "s10", "s11"]}
    
    # ��Ԃƍs���ɑ΂���J�ڐ���(�_�C�i�~�N�X)
    # ��ʓI��MDP�ɂ�����_�C�i�~�N�X�͊m��P(s'|s,a)�ŕ\����邪�A�����ł͌���_�I�ȃ_�C�i�~�N�X���̗p
    # �����珇�Ԃɍs�����͂�"left"�C"top"�C"right"�C"down"�̏ꍇ�̊e��Ԃ̑J�ڐ������
    # ��j���"s0"�̂Ƃ��A
    #          "left"���󂯎���Ă��ړ����ł��Ȃ��̂őJ�ڐ�͌��݂Ɠ���"s0"
    #          "top"���󂯎���Ă��ړ����ł��Ȃ��̂őJ�ڐ�͌��݂Ɠ���"s0"
    #          "right"���󂯎���Ă��ړ����ł��Ȃ��̂őJ�ڐ�͌��݂Ɠ���"s0"
    #          "down"���󂯎�����牺�ֈړ��ł���̂őJ�ڐ��"s1"
    # ���̑��S�Ă̏�Ԃ����l
    dynamics = {"s0":["s0", "s0", "s0", "s1"],
                "s1":["s1", "s0", "s2", "s1"],
                "s2":["s1", "s2", "s3", "s2"],
                "s3":["s2", "s4", "s6", "s7"],
                "s4":["s4", "s4", "s5", "s3"],
                "s5":["s4", "s5", "s5", "s6"],
                "s6":["s3", "s5", "s6", "s6"],
                "s7":["s7", "s3", "s7", "s8"],
                "s8":["s10", "s7", "s9", "s8"],
                "s9":["s8", "s9", "s9", "s9"],
                "s10":["s11", "s10", "s8", "s10"],
                "s11":["s11", "s11", "s10", "s11"]}
    
    def __init__(self, model=None):
        super(EasyMaze, self).__init__()

        self.fig = None
        self.ax = None
        self.state = None

        # �s����ԂƂ���0����3�܂ł�4��ނ̗��U�l��ΏۂƂ���
        # ���Ȃ݂ɁA0��"left"�A1��"top"�A2�́hright�h�A3��"down"�ɑΉ�������
        self.action_space = gym.spaces.Discrete(4)

        # ��Ԃ̓G�[�W�F���g�����݂���Z���̈ʒu(12���)
        self.observation_space = gym.spaces.Discrete(12)

        # ������V�̒l��0����1�̊ԂƂ���
        self.reward_range = (0, 1)

    def reset(self):
        # ���H�̃X�^�[�g�ʒu��"s0"�Ƃ���
        self.state = "s0"
        # ������Ԃ̔ԍ����ϑ��Ƃ��ĕԂ�
        return int(self.state[1:])

    def step(self, action):
        # ���݂̏�Ԃƍs�����玟�̏�ԂɑJ��
        self.state = self.dynamics[self.state][action]

        # �S�[�����"s11"�ɑJ�ڂ��Ă�����I���������Ƃ�done�Ɋi�[����V1���i�[
        # ���̑��̏�ԂȂ�done=False, reward=0�Ƃ���
        if self.state == "s11":
            done = True
            reward = 1
        else:
            done = False
            reward = 0

        # ����̗�ł�info�͎g�p���Ȃ�
        info = {}

        return int(self.state[1:]), reward, done, info

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
        self.plot_agent(self.state)
        # matplotlib�ō쐬�����}��z���RGB�z��ɕϊ�
        rgb_array = self.fig2array()[:, :, :3]
        # RGB�z������^�[��
        return rgb_array 

    # ���H��`�悷��֐�
    def make_maze(self):
        self.fig = plt.figure(figsize=(5, 5), dpi=200)
        self.ax = plt.axes()
        self.ax.axis("off")

        # ���H�̊O�g��\��
        for i in range(len(self.maze_info_rec["xy"])):
            r = patches.Rectangle(xy=self.maze_info_rec["xy"][i], 
                                  width=self.maze_info_rec["width"][i],
                                  height=self.maze_info_rec["height"][i], 
                                  color=self.maze_color,
                                  fill=True)
            self.ax.add_patch(r)

        # �_���ɂ��g�̕\��
        for i in range(len(self.maze_info_line["s_xy"])):
            self.ax.plot([self.maze_info_line["s_xy"][i][0], self.maze_info_line["e_xy"][i][0]],
                         [self.maze_info_line["s_xy"][i][1], self.maze_info_line["e_xy"][i][1]],
                         linewidth=1,
                         linestyle="--",
                         color=self.maze_color)

        # ��Ԃ̃e�L�X�g��\��(�X�^�[�g��ԂƃS�[����Ԃ͌�ŕ`��)
        for i in range(1, len(self.maze_state_pos["xy"])-1):
            self.ax.text(self.maze_state_pos["xy"][i][0], 
                         self.maze_state_pos["xy"][i][1], 
                         self.maze_state_pos["text"][i], 
                         size=14, 
                         ha="center")
        
        # �X�^�[�g��Ԃ̃e�L�X�g��`��
        self.ax.text(self.maze_state_pos["xy"][0][0], 
                         self.maze_state_pos["xy"][0][1], 
                         "s0\n start", 
                         size=14, 
                         ha="center")

        # �S�[����Ԃ̃e�L�X�g��`��
        self.ax.text(self.maze_state_pos["xy"][11][0], 
                         self.maze_state_pos["xy"][11][1], 
                         "s11\n goal", 
                         size=14, 
                         
                         ha="center")

    # �G�[�W�F���g��`��
    def plot_agent(self, state_name):
        self.clear_agent()
        # plt.pause(1)
        state_index = self.maze_state_pos["text"].index(state_name)
        agent_pos = self.maze_state_pos["xy"][state_index]
        self.agent_marker = self.ax.plot([agent_pos[0]], 
                             [agent_pos[1]],
                              marker="o",
                              color=self.agent_color,
                              markersize=50)
        
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
    
if __name__ == '__main__':
    
    env = EasyMaze()
    
    obs = env.reset()
    env.render()
    # plt.pause(3)
    env.render_agent()
    for _ in range(300):
        
        action = env.action_space.sample()
        print(action)
        obs, re, done, info = env.step(action)
        if done:
            env.reset()
        # ��Ԃ�����
        env.render_agent()
        plt.draw()
        plt.pause(0.1)  # ��Ԃ��\������鎞�ԁi�K�v�ɉ����Ē����j
    plt.pause(5)
    # import matplotlib.pyplot as plt

    # # ���H�����������A�ŏ��̎��o�����s��
    # env = EasyMaze()
    # obs = env.reset()
    # plt.imshow(env.render(), cmap='viridis')  # �G�[�W�F���g�̏����ʒu��\��
    # plt.show()

    # # �G�[�W�F���g�𓮂����ď�ԑJ�ڂ�����
    # for _ in range(200):
    #     action = env.action_space.sample()  # �����_���ȍs����I��
    #     obs, re, done, info = env.step(action)  # ��ԑJ�ڂ����s
    #     plt.imshow(env.render(), cmap='viridis')  # �V������Ԃ�\��
    #     plt.pause(0.1)  # 0.1�b�Ԃ̈ꎞ��~
    #     if done:
    #         obs = env.reset()  # �S�[���ɒB�����烊�Z�b�g

    # # �E�B���h�E�����
    # plt.close()
    
    
    # env = EasyMaze()
    # obs = env.reset()

    # # �`��p�̃E�B���h�E���쐬
    # plt.figure()

    # for _ in range(10):
    #     action = env.action_space.sample()
    #     obs, re, done, info = env.step(action)

    #     # ��Ԃ�����
    #     plt.clf()  # �E�B���h�E���N���A
    #     plt.imshow(env.render(), cmap='viridis')  # �V������Ԃ�\��
    #     plt.pause(0.1)  # 0.1�b�Ԃ̈ꎞ��~

    #     if done:
    #         env.reset()

    # # �E�B���h�E�����
    # plt.close()
    
