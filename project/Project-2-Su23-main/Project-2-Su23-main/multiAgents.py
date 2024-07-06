# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
            #2D STATE(walls: %, ghost: G, food: ., pac: ^), information of the score included
        newPos = successorGameState.getPacmanPosition()
            #the coordinate of position included, ex: (1,1)
        newFood = successorGameState.getFood()
            #row*column False or True
        newGhostStates = successorGameState.getGhostStates()
            #already a list
            #newGhostStates[0] = (x,y), NSEW or STOP
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
            #[0]

        "*** YOUR CODE HERE ***"

        gscore=0
        fscore=0
        ascore=0
        x,y=newPos[0],newPos[1]
        food_position_list=newFood.asList()
        ghost_position=newGhostStates[0].getPosition()    
        candidate=[]
        
        #consider food location
        for i in food_position_list:
            ix,iy=i[0],i[1]
            length_f=abs(ix-x)+abs(iy-y)
            candidate.append(length_f)
        if candidate:
            fscore=min(candidate)

        #consider ghost location
        gx,gy=ghost_position[0],ghost_position[1]
        gscore=abs(gx-x)+abs(gy-y)

        #default ghosts are random. so, dangerous pessimism must lead to low score.
        #let's be happier if ghost is far away! that is, weight more to the gscore
            #pacman wins, but the score becomes MINUS.
            #WHY? maybe too much time
        #the pacman keeps dieing at the trial 2. the reason was dangerous optimism
        #let's weight less to the gscore
            #the pacman at the trial 4 stops for a long time, leading to the MINUS score.
        #let's weight more to ascore
            #ascore = 50
        
        if action == "Stop":
            ascore=10

        #return successorGameState.getScore()+gscore-fscore-ascore
        return successorGameState.getScore()+0.5*gscore-fscore-ascore

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        /////////////////////////////////////
        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        /////////////////////////////////////
        """
        "*** YOUR CODE HERE ***"
        
        idx=0
        depth=0
        return self.value_function(gameState, idx, depth)[1] #[0]: value, [1]: action

        
        util.raiseNotDefined()


    def value_function(self, gameState: GameState, idx, depth):
        '''
        if the state is terminal: return the state's utility
        if the next agent is MAX: return max-value(state) ==> pacman is a maximizer
        if the next agent is MIN: return min-value(state) ==> ghost1, ghost2, ghost3... are minimizers
        
        '''
        if self.depth == depth or not gameState.getLegalActions(idx) :
            return self.evaluationFunction(gameState),None
        
        
        if idx == 0:  #pacman is a maximizer
            return self.max_value(gameState, idx, depth)
        else:  #ghost1, ghost2, ghost3... are minimizers
            return self.min_value(gameState, idx, depth)

        
        
    def max_value(self, gameState: GameState, idx, depth):
        '''
        initialize v ==> -infinity 10e100
        for each successor of state: #one by one (successor)==> for i in branches(action), generate successor, and then compare!
            v=max(v,value(successor))
        return v
        '''

        v=-10e100
        v_act=None
        successor_depth = depth
        
        for i in gameState.getLegalActions(idx):
            successor_i = gameState.generateSuccessor(idx, i) #successor, after agent takes an legal action i.
            if idx+1 - gameState.getNumAgents() == 0:
                successor_idx = 0 #pacman again, and the depth increases by 1
                successor_depth = depth+1
            else:
                successor_idx = idx + 1


            successor_value = self.value_function(successor_i, successor_idx, successor_depth)

            if successor_value[0]-v>0:
                v=successor_value[0]
                v_act = i

        return v,v_act
    
    
    def min_value(self, gameState: GameState, idx, depth):
        '''
        initialize v ==> infinity 10e100
        for each successor of state:
            v=min(v,value(successor))
        return v
        '''
        v=10e100
        v_act=None
        successor_depth = depth
        
        for i in gameState.getLegalActions(idx):
            successor_i = gameState.generateSuccessor(idx, i) #successor, after agent takes an legal action i.
            if idx+1 - gameState.getNumAgents() == 0:
                successor_idx = 0 #pacman again, and the depth increases by 1
                successor_depth = depth+1
            else:
                successor_idx = idx + 1


            successor_value = self.value_function(successor_i, successor_idx, successor_depth)

            if v-successor_value[0]>0:
                v=successor_value[0]
                v_act = i

        return v,v_act
    

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        idx=0
        depth=0
        alpha=-10e100
        beta=10e100
        return self.value_function(gameState, idx, depth,alpha,beta)[1] #[0]: value, [1]: action
        util.raiseNotDefined()


    def value_function(self, gameState: GameState, idx, depth,alpha,beta):
        '''
        if the state is terminal: return the state's utility
        if the next agent is MAX: return max-value(state) ==> pacman is a maximizer
        if the next agent is MIN: return min-value(state) ==> ghost1, ghost2, ghost3... are minimizers
        
        '''
        if self.depth == depth or not gameState.getLegalActions(idx) :
            return self.evaluationFunction(gameState),None
        
        
        if idx == 0:  #pacman is a maximizer
            return self.max_value(gameState, idx, depth,alpha,beta)
        else:  #ghost1, ghost2, ghost3... are minimizers
            return self.min_value(gameState, idx, depth,alpha,beta)

        
        
    def max_value(self, gameState: GameState, idx, depth,alpha,beta):
        '''
        initialize v ==> -infinity 10e100
        for each successor of state: #one by one (successor)==> for i in branches(action), generate successor, and then compare!
            v=max(v,value(successor))
        return v
        '''

        v=-10e100
        v_act=None
        successor_depth = depth
        
        for i in gameState.getLegalActions(idx):
            successor_i = gameState.generateSuccessor(idx, i) #successor, after agent takes an legal action i.
            if idx+1 - gameState.getNumAgents() == 0:
                successor_idx = 0 #pacman again, and the depth increases by 1
                successor_depth = depth+1
            else:
                successor_idx = idx + 1


            successor_value = self.value_function(successor_i, successor_idx, successor_depth,alpha,beta)

            if successor_value[0]>v:
                v=successor_value[0]
                v_act = i
                if v>beta:
                    return v, v_act
                alpha = max(alpha,v)
            else:
                if v>beta:
                    return v, v_act
                alpha = max(alpha,v)
                

        return v,v_act
    
    
    def min_value(self, gameState: GameState, idx, depth,alpha,beta):
        '''
        initialize v ==> infinity 10e100
        for each successor of state:
            v=min(v,value(successor))
        return v
        '''
        v=10e100
        v_act=None
        successor_depth = depth
        
        for i in gameState.getLegalActions(idx):
            successor_i = gameState.generateSuccessor(idx, i) #successor, after agent takes an legal action i.
            if idx+1 - gameState.getNumAgents() == 0:
                successor_idx = 0 #pacman again, and the depth increases by 1
                successor_depth = depth+1
            else:
                successor_idx = idx + 1


            successor_value = self.value_function(successor_i, successor_idx, successor_depth,alpha,beta)

            if successor_value[0]<v:
                v=successor_value[0]
                v_act = i
                if v<alpha:
                    return v, v_act
                beta = min(beta,v)
            else:
                if v<alpha:
                    return v, v_act
                beta = min(beta,v)

        return v,v_act

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        idx=0
        depth=0
        return self.value_function(gameState, idx, depth)[1] #[0]: value, [1]: action


        util.raiseNotDefined()

    def value_function(self, gameState: GameState, idx, depth):
        '''
        if the state is terminal: return the state's utility
        if the next agent is MAX: return max-value(state) ==> pacman is a maximizer
        if the next agent is MIN: return min-value(state) ==> ghost1, ghost2, ghost3... are minimizers
        
        '''
        if self.depth == depth or not gameState.getLegalActions(idx) :
            return self.evaluationFunction(gameState),None
        
        
        if idx == 0:  #pacman is a maximizer
            return self.max_value(gameState, idx, depth)
        else:  #ghost1, ghost2, ghost3... are minimizers
            return self.exp_value(gameState, idx, depth)

        
        
    def max_value(self, gameState: GameState, idx, depth):
        '''
        initialize v ==> -infinity 10e100
        for each successor of state: #one by one (successor)==> for i in branches(action), generate successor, and then compare!
            v=max(v,value(successor))
        return v
        '''

        v=-10e100
        v_act=None
        successor_depth = depth
        
        for i in gameState.getLegalActions(idx):
            successor_i = gameState.generateSuccessor(idx, i) #successor, after agent takes an legal action i.
            if idx+1 - gameState.getNumAgents() == 0:
                successor_idx = 0 #pacman again, and the depth increases by 1
                successor_depth = depth+1
            else:
                successor_idx = idx + 1


            successor_value = self.value_function(successor_i, successor_idx, successor_depth)

            if successor_value[0]-v>0:
                v=successor_value[0]
                v_act = i

        return v,v_act
    
    
    def exp_value(self, gameState: GameState, idx, depth):
        '''
        initialize v ==> infinity 10e100
        for each successor of state:
            v=min(v,value(successor))
        return v
        '''
        v=0
        v_act=None
        successor_depth = depth
        
        for i in gameState.getLegalActions(idx):
            successor_i = gameState.generateSuccessor(idx, i) #successor, after agent takes an legal action i.
            if idx+1 - gameState.getNumAgents() == 0:
                successor_idx = 0 #pacman again, and the depth increases by 1
                successor_depth = depth+1
            else:
                successor_idx = idx + 1


            successor_value = self.value_function(successor_i, successor_idx, successor_depth)

            #probablilty of going each successor is the same: 1/branch_num
            p=0
            for i in gameState.getLegalActions(idx):
                p+=1
            
            p = 1/p
            v+=p*successor_value[0]
            v_act = i

        return v,v_act


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>

    The basic structure of betterEvaluationFunction
    is similar to the evaluationFunction of the ReflexAgent.

    However, the difference from the ReflexAgent is
    1) that no action parameter is needed,
    and 2) that it is a function applied within expectimax.

    Thus, in the case of betterEvaluationFunction,
    only information on the current position, current food, and current ghost states is needed
    because the state must be evaluated, not the action.

    In other words, the code related to the successor gamestate and action
    was not required, so was modified and removed.

    In addition, in the case of the score that is returned at the end,
    we continuously changed the weight value that is multiplied by each feature of a given state (gscore, fscore)
    and multiplied by the weight value that experimentally produced the best results.

    """
    "*** YOUR CODE HERE ***"
    currentPos = currentGameState.getPacmanPosition()
        #the coordinate of position included, ex: (1,1)
    currentFood = currentGameState.getFood()
        #row*column False or True
    currentGhostStates = currentGameState.getGhostStates()
        #already a list
        #newGhostStates[0] = (x,y), NSEW or STOP

    gscore=0
    fscore=0

    x,y=currentPos[0],currentPos[1]
    food_position_list=currentFood.asList()
    ghost_position=currentGhostStates[0].getPosition()    
    candidate=[]
    
    #consider food location
    for i in food_position_list:
        ix,iy=i[0],i[1]
        length_f=abs(ix-x)+abs(iy-y)
        candidate.append(length_f)
    if candidate:
        fscore=min(candidate)

    #consider ghost location
    gx,gy=ghost_position[0],ghost_position[1]
    gscore=abs(gx-x)+abs(gy-y)

    #return currentGameState.getScore()+0.5*gscore-fscore 
    return currentGameState.getScore()+0.4*gscore-fscore


    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
