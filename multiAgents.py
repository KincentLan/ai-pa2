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

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
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

    def evaluationFunction(self, currentGameState, action):
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
        newPos = successorGameState.getPacmanPosition()
        newGhostStates = successorGameState.getGhostStates()

        currentFoodPosses = currentGameState.getFood().asList()
        currentPos = currentGameState.getPacmanPosition()
        
        utility = 0
        closestFood, distance = currentFoodPosses[0], manhattanDistance(currentFoodPosses[0], currentPos)
                
        for food in currentFoodPosses:
            curr_distance = manhattanDistance(currentPos, food)
            if distance > curr_distance:
                closestFood, distance = food, curr_distance

        if manhattanDistance(closestFood, newPos) < distance:
            utility += 10
        
        for ghost in newGhostStates:
            ghostPos = ghost.getPosition()
            distanceToGhost = manhattanDistance(ghostPos, newPos)

            if ghostPos == newPos:
                utility -= 99999
            
            utility += min(distanceToGhost - 1, 10)
        
        return utility

def scoreEvaluationFunction(currentGameState):
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

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

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
        """
        return self.value(gameState, 0, 0)[1]
    
    def value(self, state, depth, agent):
        if state.isWin() or state.isLose() or depth == self.depth:
            return self.evaluationFunction(state), Directions.STOP
        
        if agent == 0:
            return self.max_value(state, depth, agent)

        return self.min_value(state, depth, agent)


    def max_value(self, state, depth, agent):
        legalMoves = state.getLegalActions(agent)
        values = []
        
        for action in legalMoves:
            nextState = state.generateSuccessor(agent, action)
            values.append(self.value(nextState, depth, agent + 1)[0])
        
        max_idx = max(range(len(values)), key=values.__getitem__)
        
        return values[max_idx], legalMoves[max_idx]


    def min_value(self, state, depth, agent):
        legalMoves = state.getLegalActions(agent)
        values = []
        
        for action in legalMoves:
            nextState = state.generateSuccessor(agent, action)

            if agent + 1 == state.getNumAgents():
                values.append(self.value(nextState, depth + 1, 0)[0])
            else:
                values.append(self.value(nextState, depth, agent + 1)[0])
        
        min_idx = min(range(len(values)), key=values.__getitem__)
        
        return values[min_idx], legalMoves[min_idx]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        return self.value(gameState, 0, 0)[1]
    

    def value(self, state, depth, agent):
        if state.isWin() or state.isLose() or depth == self.depth:
            return self.evaluationFunction(state), Directions.STOP
        
        if agent == 0:
            return self.max_value(state, depth, agent)

        return self.exp_value(state, depth, agent)


    def max_value(self, state, depth, agent):
        legalMoves = state.getLegalActions(agent)
        values = []
        
        for action in legalMoves:
            nextState = state.generateSuccessor(agent, action)
            values.append(self.value(nextState, depth, agent + 1)[0])
        
        max_idx = max(range(len(values)), key=values.__getitem__)
        
        return values[max_idx], legalMoves[max_idx]


    def exp_value(self, state, depth, agent):
        legalMoves = state.getLegalActions(agent)
        probability = 1 / len(legalMoves)
        value = 0
        
        for action in legalMoves:
            nextState = state.generateSuccessor(agent, action)

            if agent + 1 == state.getNumAgents():
                value += probability * self.value(nextState, depth + 1, 0)[0]
            else:
                value += probability * self.value(nextState, depth, agent + 1)[0]
        
        return value, random.choice(legalMoves)
    
def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>

    - distance away from ghosts
    - distance to scared ghosts
    - distance to closest food
    - distance to capsule
    - amount of capsules left
    - amount of food left
    """
    if currentGameState.isLose():
        return -float('inf')
    if currentGameState.isWin():
        return float('inf')
    
    currentScore = scoreEvaluationFunction(currentGameState)
    ghostStates = currentGameState.getGhostStates()
    currentFoods = currentGameState.getFood().asList()
    currentCapsules = currentGameState.getCapsules()
    currentPos = currentGameState.getPacmanPosition()    
    
    utility = currentScore - (4 * len(currentFoods)) - (20 * len(currentCapsules))
    minFoodDist = float('inf')
    minCapsuleDist = float('inf')

    for food in currentFoods:
        currentDist = manhattanDistance(currentPos, food)
        minFoodDist = min(minFoodDist, currentDist)

    for capsule in currentCapsules:
        currentDist = manhattanDistance(currentPos, capsule)
        minCapsuleDist = min(minCapsuleDist, currentDist)

    utility -= 1.5 * minFoodDist if minFoodDist < float('inf') else 0
    utility -= 4 * minCapsuleDist if minCapsuleDist < float('inf') else 0

    minGhostDist = float('inf')
    minScaredGhostDist = float('inf')

    for ghost in ghostStates:
        currentDist = manhattanDistance(currentPos, ghost.getPosition())

        if ghost.scaredTimer and minScaredGhostDist > currentDist:
            minScaredGhostDist = currentDist

        if not ghost.scaredTimer and minGhostDist > currentDist:
            minGhostDist = currentDist

    utility -= 2 * (1/minGhostDist) if minGhostDist < float('inf') else 0
    utility -= 2 * minScaredGhostDist if minScaredGhostDist < float('inf') else 0

    return utility

# Abbreviation
better = betterEvaluationFunction
