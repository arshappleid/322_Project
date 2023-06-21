# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    
    stack = util.Stack(); ## Stack to store the nodes to be visited
    parent = dict(); ## Dictionary to store the parent of each node
    steps = list(); ## List to store the steps to reach the goal state
    haveVisited = set(); ## Set to store the nodes that have been visited
    startState = (problem.getStartState(), None, None); ## Start State

    while(not stack.isEmpty()):
        state = stack.pop(); ## Get the first element from the stack
        haveVisited.add(state); ## Add the state to the visited set
        if(problem.isGoalState(state[0])):
            ## Traverse Back to the start state to get the steps
            while(state != startState):
                steps.append(state[1]);
                state = parent[state];
            steps.reverse();
            return steps;

        successors = problem.getSuccessors(state[0]);
        for successor in successors:
            if(successor[0] not in haveVisited):
                stack.push(successor);
                parent[successor] = state;
    return None; ## Goal State was not found

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""

    que = util.Queue(); ## Queue to store the nodes to be visited
    parent = dict(); ## Dictionary to store the parent of each node
    steps = list(); ## List to store the steps to reach the goal state
    haveVisited = set(); ## Set to store the nodes that have been visited

    startState = (problem.getStartState(), None, None); ## Start State
    haveVisited.add(startState); ## Add the start state to the visited set
    parent[startState] = None; ## Start State has no parent
    que.push(startState); ## Add the start state to the queue

    while(not que.isEmpty()):
        state = que.pop(); ## Get the first element from the queue

        if(problem.isGoalState(state[0])):
            ## Traverse Back to the start state to get the steps
            while(state != startState):
                steps.append(state[1]);
                state = parent[state];
            steps.reverse();
            return steps;

        successors = problem.getSuccessors(state[0]);
        for successor in successors:
            if(successor[0] not in haveVisited):
                que.push(successor);
                parent[successor] = state;
                haveVisited.add(successor[0]);  ## Visit the current state
    return None; ## Goal State was not found

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    ### BFS , but search the node with least cost first , Sort the successor by cost , or we can use a priority queue to get the node with least cost

    que = util.Queue(); ## Queue to store the nodes to be visited
    parent = dict(); ## Dictionary to store the parent of each node
    steps = list(); ## List to store the steps to reach the goal state
    haveVisited = set(); ## Set to store the nodes that have been visited

    startState = (problem.getStartState(), None, None); ## Start State
    haveVisited.add(startState); ## Add the start state to the visited set
    parent[startState] = None; ## Start State has no parent
    que.push(startState); ## Add the start state to the queue

    while(not que.isEmpty()):
        state = que.pop(); ## Get the first element from the queue

        if(problem.isGoalState(state[0])):
            ## Traverse Back to the start state to get the steps
            while(state != startState):
                steps.append(state[1]);
                state = parent[state];
            steps.reverse();
            return steps;

        successors = problem.getSuccessors(state[0]);
        successors.sort(key = lambda x: x[2]); ## Sort the successors by cost
        for successor in successors:
            if(successor[0] not in haveVisited):
                que.push(successor);
                parent[successor] = state;
                haveVisited.add(successor[0]);  ## Visit the current state
    return None; ## Goal State was not found

            
def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    priorityQue = util.PriorityQueue()
    have_visited = set()

    start_state = problem.getStartState()
    priorityQue.push((start_state, list()), heuristic(start_state, problem))

    while(not priorityQue.isEmpty()):
        state, path = priorityQue.pop()  # Take current state and path
        have_visited.add(state)  # Visit the current state

        # Found goal state
        if (problem.isGoalState(state)): return path

        successors = problem.getSuccessors(state)
        for successor in successors:

            # Already have visited the particular node
            if (successor[0] in have_visited): continue
            
            # Search to see if successor already exists in the frontier(priorityQue)
            frontier_exists = False
            for element in priorityQue.heap:
                if (successor[0] == element[2][0]):
                    frontier_exists = True
                    break
            
            heuristicValue = heuristic(successor[0], problem)  # heuristic cost

            newPath = path + [successor[1]]
            newPriority = problem.getCostOfActions(newPath)

            # State does not exist either in searched state nor in the frontier, insert it
            if (not frontier_exists):
                priorityQue.push((successor[0], newPath), newPriority + heuristicValue)
            
            # Successor exists in the frontier with a higher path cost - update its path cost
            elif (problem.getCostOfActions(element[2][1]) > newPriority):
                priorityQue.update((successor[0], newPath), newPriority + heuristicValue)
    
    return None;    ## Goal State was not found


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
