'''
  แก้ code และเพิ่มเติมได้ใน class YourTeamAgent เท่านั้น 
  ตอนส่งไฟล์ ให้แน่ใจว่า YourTeamAgent ไม่มี error และ run ได้
  ส่งแค่ submission.py ไฟล์เดียว
'''
from util import manhattanDistance,PriorityQueue
from game import Directions
import random, util,copy
from typing import Any, DefaultDict, List, Set, Tuple
import time;
from game import Agent
from pacman import GameState
from game import Actions
import heapq
from typing import List, Tuple, Deque
from collections import deque

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """
  def __init__(self):
    self.lastPositions = []
    self.dc = None


  def getAction(self, gameState: GameState,agentIndex=0) -> str:
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East}
    ------------------------------------------------------------------------------
    Description of GameState and helper functions:

    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes. In this function, the |gameState| argument
    is an object of GameState class. Following are a few of the helper methods that you
    can use to query a GameState object to gather information about the present state
    of Pac-Man, the ghosts and the maze.

    gameState.getLegalActions(agentIndex):
        Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

    gameState.generateSuccessor(agentIndex, action):
        Returns the successor state after the specified agent takes the action.
        Pac-Man is agent 0 and agent 1.

    gameState.getPacmanState(agentIndex):
        Returns an AgentState object for pacman (in game.py)
        state.configuration.pos gives the current position
        state.direction gives the travel vector

    gameState.getNumAgents():
        Returns the total number of agents in the game

    gameState.getScore(agentIndex):
        Returns the score of agentIndex (0 or 1) corresponding to the current state of the game

    gameState.getScores():
        Returns all the scores of the agents in the game as a list where first score corresponds to agent 0
    
    gameState.getFood():
        Returns the food in the gameState

    gameState.getPacmanPosition(agentIndex):
        Returns the pacman (agentIndex 0 or 1) position in the gameState

    gameState.getCapsules():
        Returns the capsules in the gameState

    The GameState class is defined in pacman.py and you might want to look into that for
    other helper methods, though you don't need to.
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions(agentIndex)
    gameState.getScaredTimes(agentIndex)

    # print(legalMoves)
    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action,agentIndex) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best


    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState: GameState, action: str,agentIndex=0) -> float:
    """
    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action,agentIndex)
    newPos = successorGameState.getPacmanPosition(agentIndex)
    oldFood = currentGameState.getFood()

    return successorGameState.getScore(agentIndex)


def scoreEvaluationFunction(currentGameState: GameState,agentIndex=0) -> float:
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore(agentIndex)

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

  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '8',agentIndex=0):
    self.index = agentIndex 
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

######################################################################################
# implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Minimax agent
  """

  def getAction(self, gameState: GameState,agentIndex = 0) -> str:
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. 

      Here are some method calls that might be useful when implementing minimax.

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent (0 or 1) takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game

      gameState.getScore(agentIndex):
        Returns the score of agentIndex (0 or 1) corresponding to the current state of the game

      gameState.getLegalActions(agentIndex):
        Returns the legal actions for the agent specified (0 or 1). Returns Pac-Man's legal moves by default.

      gameState.getPacmanState(agentIndex):
          Returns an AgentState (0 or 1) object for pacman (in game.py)
          state.configuration.pos gives the current position
          state.direction gives the travel vector

      gameState.getNumAgents():
          Returns the total number of agents in the game

      gameState.getScores():
          Returns all the scores of the agents in the game as a list where first score corresponds to agent 0
      
      gameState.getFood():
          Returns the food in the gameState

      gameState.getPacmanPosition(agentIndex):
          Returns the pacman (agentIndex = 0 or 1) position in the gameState

      gameState.getCapsules():
          Returns the capsules in the gameState

      self.depth:
        The depth to which search should continue

    """
    self.index = agentIndex
    bestVal = -float('inf')
    bestAction = None
    scoreStart = copy.deepcopy(gameState.getScores())
    legalMoves = gameState.getLegalActions(agentIndex)
    
    if len(legalMoves) == 1:
      return legalMoves[0]
    else: 
      for action in legalMoves:
        successorGameState = gameState.generateSuccessor(agentIndex, action)
        val = self.minimax(successorGameState,(agentIndex+1)%2,self.depth-1)
        if val > bestVal:
          bestVal = val
          bestAction = action
      # print("score ",gameState.getScore(self.index))
      # print("score ",gameState.getScores())
      return bestAction

  def minimax(self,gameState: GameState,agentIndex,depth):
    if gameState.isWin() or gameState.isLose() or depth == 0:
      return self.evaluationFunction(gameState,agentIndex)
    
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions(agentIndex)
    # print(legalMoves)
    if agentIndex == self.index:
      best = -float('inf')
      for action in legalMoves:
        successorGameState = gameState.generateSuccessor(agentIndex, action)
        best = max(best,self.minimax(successorGameState,(agentIndex+1)%2,depth-1))
      return best
    else:
      best = float('inf')
      for action in legalMoves:
        successorGameState = gameState.generateSuccessor(agentIndex, action)
        best = min(best,self.minimax(successorGameState,(agentIndex+1)%2,depth-1))
      return best

  def evaluationFunction(self, currentGameState: GameState, action: str,agentIndex=0) -> float:
    """
    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action,agentIndex)
    newPos = successorGameState.getPacmanPosition(agentIndex)
    oldFood = currentGameState.getFood()

    return successorGameState.getScore()
class YourTeamAgent2(MultiAgentSearchAgent):
    """
    Your team agent
    This class makes Pac-Man find all closest capsules and food in the game state.
    If there is no food or capsules left, it makes Pac-Man randomly walk.
    Pac-Man also avoids walls and uses BFS for pathfinding.
    Additionally, if Pac-Man is scared and within a close distance of the other AI, it avoids the other AI.
    """

    def getAction(self, gameState: GameState, agentIndex: int = 0) -> str:
        """
        Returns the best action for Pac-Man to take in the given game state to minimize the total distance to the closest capsule or food.
        If there is no food or capsules left, returns a random legal action.
        If Pac-Man is scared and within a close distance of the other AI, returns an action to avoid the other AI.

        Args:
            gameState: The current game state.
            agentIndex: The index of the agent (0 for Pac-Man, 1 for the second agent).

        Returns:
            The optimal action to take in the given game state, prioritizing avoiding the other AI when Pac-Man is scared and close to the enemy AI.
        """
        # Get the legal actions for Pac-Man
        legalActions = gameState.getLegalActions(agentIndex)

        # Get Pac-Man's current position
        pacmanPosition = gameState.getPacmanPosition(agentIndex)

        # Get the list of remaining capsules and food in the game state
        capsules = gameState.getCapsules()
        foodGrid = gameState.getFood()
        foodList = foodGrid.asList()

        # Get the state of the other agent (enemy AI)
        enemyAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
        enemyAI = gameState.getPacmanState(enemyAgentIndex)
        enemyPosition = gameState.getPacmanPosition(enemyAgentIndex)
        enemyScaredTimes = enemyAI.scaredTimer

        # Get the scared times for our AI
        scaredTimes = gameState.getScaredTimes(agentIndex)

        # Calculate the distance from Pac-Man to the other AI
        distanceToEnemyAI = manhattanDistance(pacmanPosition, enemyPosition)

        # Check if our AI is scared and the distance to the other AI is less than 3
        if scaredTimes > 0 and distanceToEnemyAI < 3:
            # If our AI is scared and close to the enemy AI, avoid the other AI
            return self.findBestActionToAvoidOtherAI(gameState, agentIndex, pacmanPosition, enemyPosition)

        # Check if there is no food and no capsules left
        if not capsules and not foodList:
            # If there is no food and no capsules left, return a random legal action
            return random.choice(legalActions)

        # Determine the closest target (capsule or food)
        if capsules:
            closestTarget = min(capsules, key=lambda capsule: manhattanDistance(pacmanPosition, capsule))
        else:
            closestTarget = min(foodList, key=lambda food: manhattanDistance(pacmanPosition, food))

        # Find the best action to navigate towards the closest target
        bestAction = self.findBestActionToTarget(gameState, agentIndex, pacmanPosition, closestTarget)
        return bestAction

    def findBestActionToAvoidOtherAI(self, gameState: GameState, agentIndex: int, pacmanPosition: Tuple[int, int], enemyPosition: Tuple[int, int]) -> str:
        """
        Finds the best action to maximize the distance from the other AI when Pac-Man is scared and close to the enemy AI.

        Args:
            gameState: The current game state.
            agentIndex: The index of the agent (0 for Pac-Man, 1 for the second agent).
            pacmanPosition: Pac-Man's current position.
            enemyPosition: The other AI's current position.

        Returns:
            The best action to maximize the distance from the other AI.
        """
        legalActions = gameState.getLegalActions(agentIndex)
        maxDistance = -float('inf')
        bestAction = None

        # Iterate over the legal actions and calculate the distance from the other AI
        for action in legalActions:
            # Generate the successor state after taking the action
            successorGameState = gameState.generateSuccessor(agentIndex, action)
            
            # Get Pac-Man's new position after taking the action
            newPosition = successorGameState.getPacmanPosition(agentIndex)
            
            # Calculate the Manhattan distance from the other AI
            distanceFromAI = manhattanDistance(newPosition, enemyPosition)
            
            # Update the best action and maximum distance if the current distance is greater
            if distanceFromAI > maxDistance:
                maxDistance = distanceFromAI
                bestAction = action
        
        # Return the best action that maximizes the distance from the other AI
        return bestAction

    def findBestActionToTarget(self, gameState: GameState, agentIndex: int, startPosition: Tuple[int, int], targetPosition: Tuple[int, int]) -> str:
        """
        Uses BFS to find the best action to navigate from startPosition to targetPosition, avoiding walls.

        Args:
            gameState: The current game state.
            agentIndex: The index of the agent (0 for Pac-Man, 1 for the second agent).
            startPosition: The starting position of the agent.
            targetPosition: The target position to navigate towards.

        Returns:
            The best action to take to navigate from startPosition to targetPosition.
        """
        # Get the walls of the maze
        walls = gameState.getWalls()

        # Perform BFS to find the shortest path from startPosition to targetPosition
        queue: Deque[Tuple[Tuple[int, int], List[str]]] = deque([(startPosition, [])])
        visited = set()
        visited.add(startPosition)

        # Define the possible directions and their respective actions
        directions = [
            ((0, 1), Directions.NORTH),  # North
            ((0, -1), Directions.SOUTH), # South
            ((-1, 0), Directions.WEST),  # West
            ((1, 0), Directions.EAST)   # East
        ]

        while queue:
            currentPosition, actions = queue.popleft()
            
            # If the current position is the target position, return the first action in the path
            if currentPosition == targetPosition:
                return actions[0] if actions else Directions.STOP
            
            # Explore the possible directions
            for (dx, dy), direction in directions:
                newPosition = (currentPosition[0] + dx, currentPosition[1] + dy)
                
                # If the new position is within bounds, not a wall, and not visited
                if (0 <= newPosition[0] < gameState.data.layout.width and
                    0 <= newPosition[1] < gameState.data.layout.height and
                    not walls[newPosition[0]][newPosition[1]] and
                    newPosition not in visited):
                    # Mark the new position as visited and enqueue it
                    visited.add(newPosition)
                    queue.append((newPosition, actions + [direction]))

        # If there is no path found, return a random legal action as a fallback
        return random.choice(gameState.getLegalActions(agentIndex))

    def evaluationFunction(self, currentGameState: GameState, agentIndex: int) -> float:
        """
        Evaluation function for the current game state.

        This function evaluates the game state by considering the distance to the closest capsule or food.
        Lower values indicate a better state (closer to the target).

        Args:
            currentGameState: The current game state.
            agentIndex: The index of the agent (0 for Pac-Man).

        Returns:
            A float value representing the utility of the game state.
        """
        # Get Pac-Man's current position
        pacmanPosition = currentGameState.getPacmanPosition(agentIndex)

        # Get the list of remaining capsules and food in the game state
        capsules = currentGameState.getCapsules()
        foodGrid = currentGameState.getFood()
        foodList = foodGrid.asList()

        # Check if there is no food and no capsules left
        if not capsules and not foodList:
            # If there is no food and no capsules left, return a high value since there are no targets
            return float('inf')

        # Determine the closest target (capsule or food)
        if capsules:
            closestTarget = min(capsules, key=lambda capsule: manhattanDistance(pacmanPosition, capsule))
        else:
            closestTarget = min(foodList, key=lambda food: manhattanDistance(pacmanPosition, food))

        # Calculate the distance to the closest target (capsule or food)
        closestTargetDistance = manhattanDistance(pacmanPosition, closestTarget)

        # Return the negative of the closest target distance for minimization
        return -closestTargetDistance
######################################################################################
class YourTeamAgent(MultiAgentSearchAgent):
    def __init__(self):
        self.lastPositions = []
        self.dc = None

    def getAction(self, gameState: GameState, agentIndex=0) -> str:
        """
        Choose the best action according to the evaluation function.
        """
        legalMoves = gameState.getLegalActions(agentIndex)
        gameState.getScaredTimes(agentIndex)

        scores = [self.evaluationFunction(gameState, action, agentIndex) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        return legalMoves[chosenIndex]

    def isAccessible(self, start, end, gameState):
        """
        Check if the end position is accessible from the start position in the game state.

        Args:
        - start: The starting position.
        - end: The ending position to check accessibility to.
        - gameState: The current state of the game.

        Returns:
        - True if the end position is accessible from the start position, False otherwise.
        """

        # A* algorithm for pathfinding
        frontier = PriorityQueue()
        frontier.push((start, []), 0)
        explored = set()

        while not frontier.isEmpty():
            current, path = frontier.pop()

            if current == end:
                # End position is reachable
                return True

            if current in explored:
                continue

            walls = gameState.getWalls()
            width, height = walls.width, walls.height

            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    next_pos = (current[0] + dx, current[1] + dy)
                    # Check if next_pos is within the boundaries of the maze and not a wall
                    if 0 <= next_pos[0] < width and 0 <= next_pos[1] < height and not walls[next_pos[0]][next_pos[1]]:
                        if gameState.hasWall(next_pos[0], next_pos[1]):
                            continue

                        if next_pos not in explored:
                            frontier.push((next_pos, path + [next_pos]), len(path) + 1)

            explored.add(current)

        # End position is not reachable
        return False

    def evaluationFunction(self, currentGameState: GameState, action: str, agentIndex=0, alpha=float('-inf'),
                           beta=float('inf')) -> float:
        """
        Evaluate the current game state based on various factors and return a score.
        Higher scores indicate better states.

        Args:
        - currentGameState: The current state of the game.
        - action: The action to evaluate.
        - agentIndex: The index of the agent (0 for player 1, 1 for player 2).
        - alpha: The alpha value for alpha-beta pruning (default is negative infinity).
        - beta: The beta value for alpha-beta pruning (default is positive infinity).

        Returns:
        - The score of the evaluated state.
        """

        successorGameState = currentGameState.generateSuccessor(agentIndex, action)

        new_agent_pos = successorGameState.getPacmanPosition(agentIndex)

        old_food = currentGameState.getFood()
        new_food = successorGameState.getFood()

        num_food_collected = len(old_food.asList()) - len(new_food.asList())

        range_to_search = 55 # Search range covers the entire map

        # Calculate the number of nearby food pellets
        nearby_food = [(x, y) for x in
                       range(new_agent_pos[0] - range_to_search, new_agent_pos[0] + range_to_search + 1)
                       for y in
                       range(new_agent_pos[1] - range_to_search, new_agent_pos[1] + range_to_search + 1)
                       if 0 <= x < new_food.width and 0 <= y < new_food.height and new_food[x][y]]

        # Calculate the number of nearby capsules
        nearby_capsules = [(x, y) for x in
                           range(new_agent_pos[0] - range_to_search, new_agent_pos[0] + range_to_search + 1)
                           for y in
                           range(new_agent_pos[1] - range_to_search, new_agent_pos[1] + range_to_search + 1)
                           if 0 <= x < new_food.width and 0 <= y < new_food.height and (x, y) in currentGameState.getCapsules()]

        # Calculate the distance to the nearest food pellet
        closest_food_distance = min(util.manhattanDistance(new_agent_pos, food) for food in nearby_food) if nearby_food else float(
            'inf')

        # Calculate the distance to the nearest capsule
        closest_capsule_distance = min(util.manhattanDistance(new_agent_pos, capsule) for capsule in nearby_capsules) if nearby_capsules else float(
            'inf')

        final_food_bonus = 20 if len(new_food.asList()) <= 1 else 10

        remaining_capsules = len(successorGameState.getCapsules())
        score = (
                successorGameState.getScore(agentIndex) +
                num_food_collected -
                remaining_capsules * 100 -
                closest_food_distance +
                final_food_bonus -
                len(new_food.asList()) * 50
        )

        # Prioritize going towards capsules
        score -= closest_capsule_distance * 10 if closest_capsule_distance != float('inf') else 0

        # Penalize actions that lead to positions with walls
        if currentGameState.hasWall(new_agent_pos[0], new_agent_pos[1]):
            score -= 1000  # Penalize heavily for hitting a wall

        # Alpha-beta pruning
        if agentIndex == 0:
            # Maximizer
            if score >= beta:
                return score
            alpha = max(alpha, score)
        else:
            # Minimizer
            if score <= alpha:
                return score
            beta = min(beta, score)

        return score






















