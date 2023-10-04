'''
ReDeveloped in 2022/07/01
@author: Hitoshi Kono, Tokyo Polytechnic University
Following packages are needed for running.
matplotlib	current:3.4.2
numpy	current:1.21.0
pygame	current:1.9.6
'''
import time
import threading
import logging
import pygame
import numpy
import random
import csv
import datetime
import matplotlib.pyplot as plt
import matplotlib.gridspec as gsc

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(asctime)s: %(threadName)s: %(message)s')

class Vars:

    # Waiting time [sec] in each action (step).
    T_STEP = 0.0

    # System variables' declaration.
    L_RATE = []     # Learning rate
    D_RATE = []     # Discount rate
    FINISH = []     # End episode number
    BOLTZMANN = []  # Temperature parameter of boltzmann function
    P_REWARD = []   # Reward of positive
    N_REWARD = []   # Reward of negative
    R_PER_STEP = 0.0 # Reward per each step (optional)

    # Set reinforcement learning parameters.
    L_RATE.append(0.1)
    D_RATE.append(0.99)
    FINISH.append(300)
    BOLTZMANN.append(0.05)
    P_REWARD.append(1.0)
    N_REWARD.append(-0.01)

    # Declaration of grid world structure with numpy.array
    START = [10, 1] # (x, y)
    GOALP = [1, 10]
    GRID = numpy.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
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

    # Constant values for the PyGame. Do not change.
    CS = 25
    SCR_X = GRID.shape[1] * CS  # column
    SCR_Y = GRID.shape[0] * CS  # row
    SCR_RECT = pygame.Rect(0, 0, SCR_X, SCR_Y)  # Rect(left,top,width,height)
    ROAD = 0  # Identification number of Road in the world
    WALL = 1  # Identification number of Wall or Obstacles in the world
    GOAL = 2  # Identification number of Goal in the world
    AGNT = 3  # Identification number of Agent in the world

    # Simulation control parameters.
    DRAW = True
    RUNNING = False
    DATE = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


class Learning:

    def __init__(self):
        # Variables and lists for logging of learning process.
        self.NEPISODE = 1   # Do not modify
        self.NSTEP = 0      # Do not modify
        self.TREWARD = 0     # Do not modify
        self.EPISODES = []
        self.STEPS = []
        self.TREWARDS = []

        # Memory space declaration for Q-table.
        self.Q = numpy.zeros((Vars.GRID.shape[1], Vars.GRID.shape[0], 5))

        self.STATE = [Vars.START[0], Vars.START[1]]       # (x, y)
        self.OLDSTATE = [Vars.START[0], Vars.START[1]]    # Old coordinates
        # List of ACT means as follow
        # ACT[0] is current executing action number
        # ACT[1] is latest executed action number
        # ACT[2] is flag of that agent can not move
        self.ACT = [0, 0, 0]

    # Execution of coordinates based on moving.
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
        elif actNum == 4:  # Stay
            return 0
        else:  # Stay
            return 0

    '''
    Selecting of an action using Q-table based on boltzmann distribution model.
    x is self.state[0]
    y is self.state[1]
    '''
    def action(self, state, oState, q, act):
        total = ret = pmin = pmax = 0.0
        v = p = numpy.zeros(5)  # List of action values
        act[1] = act[0]         # Storing executed action to old action
        rand = random.uniform(0.0, 1.0)
        boltz = Vars.BOLTZMANN[0]

        for i in range(0, 5):
            v[i] = q[state[0]][state[1]][i]
        for j in range(0, 5):
            total = total + numpy.exp(v[j] / boltz)
        for k in range(0, 5):
            p[k] = (numpy.exp(v[k] / boltz)) / total

        for n in range(0, 5):
            pmax += p[n]
            if pmin <= rand < pmax:
                ret = self.collisionCheck(state, oState, n)
                act[0] = n
            pmin += p[n]

        if ret < 0:
            act[2] = ret
        else:
            act[2] = 0

    def argMaxQ(self, state, q, numAct):
        tmpValue = numpy.zeros(numAct)
        for i in range(0, numAct):
            tmpValue[i] = q[state[0]][state[1]][i]
        return numpy.argmax(tmpValue)

    # Updating the Q-value
    def updtQ(self, state, oState, q, mode, act, reward):
        maxQact = self.argMaxQ(state, q, 5)
        TDerror = Vars.D_RATE[mode] * q[state[0]][state[1]][maxQact] - \
                  q[oState[0]][oState[1]][act[0]]
        q[oState[0]][oState[1]][act[0]] = q[oState[0]][oState[1]][act[0]] + \
                                          Vars.L_RATE[mode] * (reward + TDerror)


class Agent(threading.Thread):

    def __init__(self):
        super(Agent, self).__init__()

    agt1 = Learning()  # Agent (instance) declaration, you can add the any agent in this area.

    # Deployment (reset) the agent and goal position in the world.
    def resetWorld(self, tmpState):
        Vars.GRID[Vars.START[1]][Vars.START[0]] = Vars.AGNT
        Vars.GRID[Vars.GOALP[1]][Vars.GOALP[0]] = Vars.GOAL
        tmpState[:] = Vars.START  # Copy the default position

    '''
    This method composes the episodes (e.g. [1,2,...,n]) and the steps (e.g. [y1,y2,...yn]).
    First array's structure is two rows matrix, so second statement is processed it
    to transposed matrix like bellow:
    1, y1
    2, y2
    3, y3
    4, y4
    -
    n, yn
    '''
    def logStepEpisode(self, filename, episodes, steps, rewards):
        tmpSteps = numpy.array([episodes, steps, rewards])  # Composing two lists to one array
        tmpSteps = tmpSteps.T                               # Transposition with NumPy
        writer = csv.writer(filename)
        writer.writerows(tmpSteps)

    def logQtable(self, filename, qtable):
        numpy.savez(filename, savedQtable=qtable)

    # Execution of learning process.
    def execute(self, agt,  num):
        while agt.NEPISODE <= Vars.FINISH[num]:
            if Vars.RUNNING:
                agt.action(agt.STATE, agt.OLDSTATE, agt.Q, agt.ACT)
                agt.NSTEP += 1
                if agt.STATE == Vars.GOALP:
                    agt.updtQ(agt.STATE, agt.OLDSTATE, agt.Q, num, agt.ACT, Vars.P_REWARD[num])
                    agt.STEPS.append(agt.NSTEP)        # Append to list the number of steps
                    agt.EPISODES.append(agt.NEPISODE)  # Append to list the number of episodes
                    agt.NEPISODE += 1                  # Add one to number of episodes
                    agt.NSTEP = 0                      # Set default value as 0 step
                    agt.TREWARD = agt.TREWARD + Vars.P_REWARD[num]  # Final sum of the goal reward
                    agt.TREWARDS.append(agt.TREWARD)   # Append to list the total reward
                    agt.TREWARD = 0
                    self.resetWorld(agt.STATE)         # Reset the coordinates
                elif agt.ACT[2] < 0:
                    agt.updtQ(agt.STATE, agt.OLDSTATE, agt.Q, num, agt.ACT, Vars.N_REWARD[num])
                    agt.TREWARD += Vars.N_REWARD[num]
                else:
                    agt.updtQ(agt.STATE, agt.OLDSTATE, agt.Q, num, agt.ACT, Vars.R_PER_STEP)
                    agt.TREWARD = agt.TREWARD + Vars.R_PER_STEP
                time.sleep(Vars.T_STEP)
            else:
                time.sleep(0.1)  # Sleeping time for start

    # Described following are execution codes in the thread.
    def run(self):
        logging.info('Agent thread start')
        self.resetWorld(Vars.START)  # Deploying the agent and goal.
        fileStepsRL = open("./source/steps_" + Vars.DATE + ".csv", 'w', newline='')
        fileQtableRL = "./source/qtable_" + Vars.DATE
        logging.info('Reinforcement learning (Source task) start')
        self.execute(self.agt1, 0)
        logging.info('Reinforcement learning is terminated')
        self.logStepEpisode(fileStepsRL, self.agt1.EPISODES, self.agt1.STEPS, self.agt1.TREWARDS)
        self.logQtable(fileQtableRL, self.agt1.Q)
        logging.info('Agent thread stop')


class View(threading.Thread):

    def __init__(self):
        super(View, self).__init__()
        pygame.init()
        self.screen = pygame.display.set_mode(Vars.SCR_RECT.size)
        pygame.display.set_caption(u"Environmental view")
        self.font = pygame.font.SysFont("timesnewroman", 20)
        self.reflect(self.screen)  # First drawing of grid world
        self.screen.fill((255, 255, 255))
        self.stopEvent = threading.Event()
        self.setDaemon(True)

    def stopThread(self):
        self.stopEvent.set()

    # Value function method based on Q-value for pygame.
    def maxQvalue(self, state, q, numAct):
        tmpMaxValue = []
        for i in range(0, numAct):
            tmpMaxValue.append(q[state[0]][state[1]][i])
        return max(tmpMaxValue)

    # This function represent grid world information for display.
    def reflect(self, screen):
        for x in range(0, Vars.GRID.shape[1]):
            for y in range(0, Vars.GRID.shape[0]):
                tmpRect = pygame.Rect(x * Vars.CS, y * Vars.CS, Vars.CS, Vars.CS)
                if Vars.GRID[y][x] == Vars.WALL:
                    pygame.draw.rect(screen, (0, 0, 0), tmpRect)        # Painting with black
                elif Vars.GRID[y][x] == Vars.ROAD:
                    pygame.draw.rect(screen, (255, 255, 255), tmpRect)  # Painting with white
                    pygame.draw.rect(screen, (0, 0, 0), tmpRect, 1)     # and black lines

                    # Arrow painting based on the value function with action values.
                    actionValue = self.maxQvalue([x, y], Agent.agt1.Q, 5)
                    if actionValue != 0:
                        if actionValue > 1:
                            actionValue = 1  # Fitting to limit
                        if actionValue > 0:
                            actionValue = actionValue * 255.0
                            color = (255, 255 - actionValue, 255 - actionValue)
                            pygame.draw.rect(screen, color, tmpRect)
                            pygame.draw.rect(screen, (0, 0, 0), tmpRect, 1)

                elif Vars.GRID[y][x] == Vars.GOAL:
                    pygame.draw.rect(screen, (25, 135, 22), tmpRect)
                    pygame.draw.rect(screen, (0, 0, 0), tmpRect, 1)
                elif Vars.GRID[y][x] == Vars.AGNT:
                    pygame.draw.circle(screen, (0, 0, 255), \
                                       (int((x * Vars.CS) + (Vars.CS / 2)), \
                                        int((y * Vars.CS) + (Vars.CS / 2))), int(Vars.CS / 3))
                    pygame.draw.rect(screen, (0, 0, 0), tmpRect, 1)

    # Run function, following codes are executed as the thread.
    def run(self):
        logging.info('Environmental view thread start')

        while not self.stopEvent.wait(0.01):
            if Vars.DRAW:
                self.reflect(self.screen)
                pygame.display.update()
        pygame.quit()
        logging.info('Environmental view thread stop')


if __name__ == '__main__':
    logging.info('Starting thread as main program')
    TERMINATION = False

    logging.info('Graph initialization')
    graphpath = "./source/graphs_" + Vars.DATE + ".png"

    # # Declaration of subfigures with gridspec.
    # plt.ion()
    # gs = gsc.GridSpec(2, 7)
    # fig = plt.figure(figsize=(10, 7))
    # graph1 = fig.add_subplot(gs[1, 0:7])
    # graph2 = fig.add_subplot(gs[0, 0:4])
    # graph3 = fig.add_subplot(gs[0, 4:7])

    # # Setting of pyplot.
    # plt.rcParams['font.family'] = 'Times New Roman'
    # graph1.set_title("Learning curve (steps)")
    # graph1.set_ylabel("Number of steps")
    # graph1.set_xlabel("Number of episodes")
    # graph2.set_title("Learning curve (rewards)")
    # graph2.set_ylabel("Total rewards")
    # graph2.set_xlabel("Number of episodes")
    # graph3.set_title("Null")
    # graph3.set_ylabel("Null")
    # graph3.set_xlabel("Null")
    # # Adjust to fit among graph windows.
    # fig.tight_layout()
    # Loop for representation of graphs in plot window.
    MAX_NEPISODE = Vars.FINISH[0]  # Reinforcement learning

    # Selecting of the required threads.
    th1 = Agent()   # Thread of learning agent
    th2 = View()    # Thread of PyGame (2D)
    th1.start()
    th2.start()

    # PyGame control with key event function.
    pygame.init()
    while not TERMINATION:
        for event in pygame.event.get():            # Input of keyboard
            if event.type == pygame.QUIT:           # PyGame quit process
                th2.stopThread()
                TERMINATION = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:    # 'ESC' press, termination
                    th2.stopThread()
                    TERMINATION = True
                elif event.key == pygame.K_s:       # 's' press, start or stop
                    Vars.RUNNING = not Vars.RUNNING
                    print("Run") if Vars.RUNNING else print("Stop")
                elif event.key == pygame.K_t:       # 't' press, switch drawing
                    Vars.DRAW = not Vars.DRAW
                    print("Draw") if Vars.DRAW else print("Non-draw")

        # if Agent.agt1.NEPISODE >= 2:
        #     # Refresh graph of learning curve.
        #     graph1.clear()
        #     graph2.clear()
        #     graph3.clear()

        #     # Setting of labels and related function in pyplot.
        #     graph1.set_title("Learning curve (steps)")
        #     graph1.set_ylabel("Number of steps")
        #     graph1.set_xlabel("Number of episodes")
        #     graph1.set_yscale("log", nonpositive='clip')
        #     graph2.set_title("Learning curve (rewards)")
        #     graph2.set_ylabel("Cumulative rewards")
        #     graph2.set_xlabel("Number of episodes")
        #     graph3.set_title("Null")
        #     graph3.set_ylabel("Null")
        #     graph3.set_xlabel("Null")
        #     fig.tight_layout()

        #     if Agent.agt1.NEPISODE < 10:
        #         graph1.set_xlim(1, 10)
        #         graph2.set_xlim(1, 10)
        #     elif Agent.agt1.NEPISODE >= 10:
        #         graph1.set_xlim(1, (int(Agent.agt1.NEPISODE * 1.05)))
        #         graph2.set_xlim(1, (int(Agent.agt1.NEPISODE * 1.05)))
        #     graph1.set_ylim(1, max(Agent.agt1.STEPS) + 10)
        #     graph2.set_ylim(numpy.floor(min(Agent.agt1.TREWARDS) * 1.05), \
        #                     numpy.ceil(max(Agent.agt1.TREWARDS) * 1.05))

        #     graph1.plot(Agent.agt1.EPISODES, Agent.agt1.STEPS)
        #     graph2.plot(Agent.agt1.EPISODES, Agent.agt1.TREWARDS)
        #     plt.pause(.01)
        pygame.time.wait(10)

    th1.join()
    th2.join()

    # plt.savefig(graphpath)
    # plt.close()  # Close graph window
    pygame.quit()
    logging.info('All threads are terminated')