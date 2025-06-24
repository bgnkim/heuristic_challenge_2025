# Logging method for board execution
import heapq
import logging
# Library for OS environment
import os
import random
import sys
# Object-level deep copy method
from copy import deepcopy
# Random number generators
from random import randint as random_integer
# Type specification for Python code
from typing import Tuple, List, Literal

# Process information class: for memory usage tracking
from psutil import Process as PUInfo, NoSuchProcess
# Import some class definitions that implements the Settlers of Catan game.
from pyquoridor.board import Board
from pyquoridor.exceptions import GameOver, InvalidFence, InvalidMove  # InvalidFence 추가
from pyquoridor.square import MAX_COL, MAX_ROW

# Import action specifications
from action import Action, BLOCK, MOVE, FENCES_MAX
# Import some utilities
from util import print_board

#: True if the program run with 'DEBUG' environment variable.
IS_DEBUG = '--debug' in sys.argv
IS_RUN = 'fixed_evaluation' in sys.argv[0]

class GameBoard:
    """
    The game board object.
    By interacting with Board, you can expect what will happen afterward.
    """
    #: [PRIVATE] The game instance running currently. Don't access this directly in your agent code!
    _board: Board = None
    #: [PRIVATE] The current player's index.
    _current_player = 'white'
    #: [PRIVATE] The initial state of the board. Don't access this directly in your agent code!
    _initial = None
    #: [PRIVATE] The current state of the board. Don't access this directly in your agent code!
    _current = None
    _fences = []
    #: [PRIVATE] Logger instance for Board's function calls
    _logger = logging.getLogger('GameBoard')
    #: [PRIVATE] Memory usage tracker
    _process_info = None
    #: [PRIVATE] Fields for computing maximum memory usage. Don't access this directly in your agent code!
    _init_memory = 0
    _max_memory = 0
    #: [PRIVATE] Random seed generator
    _rng = random.Random(2329)

    def _initialize(self, start_with_random_fence: int = 0):
        """
        Initialize the board for evaluation. ONLY for evaluation purposes.
        [WARN] Don't access this method in your agent code.
        """
        # Initialize process tracker
        self._process_info = PUInfo(os.getpid())

        if IS_DEBUG:  # Logging for debug
            self._logger.debug('Initializing a new game board...')
        
        # Initialize a new game board
        self._board = Board()

        # Initialize board renderer for debugging purposes
        if IS_DEBUG:  # Logging for debug
            self._logger.debug('Rendered board: \n' + self._unique_game_state_identifier())
            self._logger.debug('\n' + print_board(self._board))

        # Store initial state representation
        self._initial = self._save_state()
        self._current = deepcopy(self._initial)
        self._fences = []

        # Update memory usage
        self._update_memory_usage()

    def reset_memory_usage(self):
        """
        Reset memory usage
        """
        self._init_memory = 0
        self._max_memory = 0
        self._update_memory_usage()

    def set_to_state(self, specific_state=None, is_initial: bool = False):
        """
        Restore the board to the initial state for repeated evaluation.
        :param specific_state: A state representation which the board reset to
        :param is_initial: True if this is an initial state to begin evaluation
        """
        assert specific_state is not None or not is_initial
        if specific_state is None:
            specific_state = self._initial
        if is_initial:
            self._initial = specific_state
            self._current = deepcopy(self._initial)
            self._rng.seed(hash(self._initial['state_id']))  # Use state_id as hash seed.

        # Restore the board to the given state.
        self._restore_state(specific_state)

        # Update memory usage
        self._update_memory_usage()

        if IS_DEBUG:  # Logging for debug
            self._logger.debug('State has been set as follows: \n' + self._unique_game_state_identifier())
            self._logger.debug('\n' + print_board(self._board))

    def is_game_end(self):
        """
        Check whether the given state indicate the end of the game

        :return: True if the game ends at the given state
        """
        is_game_end = self._board.game_finished()
        if IS_DEBUG:  # Logging for debug
            self._logger.debug(f'Querying whether the game ends in this state... Answer = {is_game_end}')
        return is_game_end

    def current_player(self):
        """
        Return the current player
        """
        return self._board.current_player()

    def get_state(self) -> dict:
        """
        Get the current board state

        :return: A copy of the current board state dictionary
        """
        if IS_DEBUG:  # Logging for debug
            self._logger.debug('Querying current state...')

        # Check whether the game has been initialized or not.
        assert self._current is not None, 'The board should be initialized. Did you run the evaluation code properly?'
        # Return the initial state representation as a copy.
        return deepcopy(self._current)

    def get_position(self, player: Literal['white', 'black']) -> Tuple[int, int]:
        """
        Get the current position of the given player
        :param player: The player
        :return: The current position
        """
        square = self._board.pawns[player].square
        return (square.row, square.col)

    def get_initial_state(self) -> dict:
        """
        Get the initial board state
        :return: A copy of the initial board state dictionary
        """
        if IS_DEBUG:  # Logging for debug
            self._logger.debug('Querying initial state...')

        # Check whether the game has been initialized or not.
        assert self._initial is not None, 'The board should be initialized. Did you run the evaluation code properly?'
        # Return the initial state representation as a copy.
        return deepcopy(self._initial)

    def get_applicable_moves(self, player: Literal['black', 'white'] = None) -> List[Tuple[int, int]]:
        """
        Get the list of applicable roads
        :param player: Player name. black or white. (You can ask your player ID by calling get_player_index())
        :return: A copy of the list of applicable move coordinates.
            (List of Tuple[int, int].)
        """
        if IS_DEBUG:  # Logging for debug
            self._logger.debug('Querying applicable move directions...')

        # Read all applicable positions
        player = self._board.current_player() if player is None else player
        applicable_positions = sorted([
            (square.row, square.col)
            for square in self._board.valid_pawn_moves(player, check_winner=False)
        ])

        # Update memory usage
        self._update_memory_usage()

        if IS_DEBUG:  # Logging for debug
            self._logger.debug(f'List of applicable move positions: {applicable_positions}')

        # Return applicable positions as list of tuples.
        return applicable_positions

    def get_applicable_fences(self, player: Literal['black', 'white'] = None)\
            -> List[Tuple[Tuple[int, int], Literal['horizontal', 'vertical']]]:
        """
        Get the list of applicable fences

        :param player: Player name. black or white. (You can ask your player ID by calling get_player_index())
        :return: A copy of the list of applicable fence coordinates with its orientation (horizontal or vertical).
            (List of Tuple[Tuple[int, int], str].)
        """
        if IS_DEBUG:  # Logging for debug
            self._logger.debug('Querying applicable move directions...')

        # Read all applicable positions
        player = self._board.current_player() if player is None else player
        if self._board.fences_left[player] == 0:
            if IS_DEBUG:  # Logging for debug
                self._logger.debug(f'{player} used all fences.')
            return []

        applicable_fences = []
        for r in range(MAX_ROW - 1):
            for c in range(MAX_COL - 1):
                # Pass positions whose center of a grid.
                if self._board.fence_center_grid[r, c]:
                    continue

                if not self._board.fence_exists(r, c, 'h'):
                    applicable_fences.append(((r, c), 'horizontal'))
                if not self._board.fence_exists(r, c, 'v'):
                    applicable_fences.append(((r, c), 'vertical'))

        applicable_fences = sorted(applicable_fences)

        # Update memory usage
        self._update_memory_usage()

        if IS_DEBUG:  # Logging for debug
            self._logger.debug(f'List of applicable fence positions: {applicable_fences}')

        # Return applicable positions as list of tuples.
        return applicable_fences

    def get_current_memory_usage(self):
        """
        :return: Current memory usage for the process having this board
        """
        try:
            usage = self._process_info.memory_full_info().uss
            if self._init_memory == 0:
                self._init_memory = usage
            return usage
        except NoSuchProcess:
            if self._max_memory >= 0:
                self._logger.warning('As tracking the process has been failed, '
                                     'I turned off memory usage tracking ability.')
                self._max_memory = -1
            return -1

    def get_max_memory_usage(self):
        """
        :return: Maximum memory usage for the process having this board
        """
        return max(0, self._max_memory - self._init_memory)

    def _update_memory_usage(self):
        """
        [PRIVATE] updating maximum memory usage
        """
        if self._max_memory >= 0:
            self._max_memory = max(self._max_memory, self.get_current_memory_usage())

    def number_of_fences_left(self, player: Literal['black', 'white']):
        """
        Number of fences remained as uninstalled.
        :param player: Player name to compute the number of fences unused
        :return: (int) The number of unused fences
        """
        return self._board.fences_left[player]

    def simulate_action(self, state: dict = None, *actions: Action) -> dict:
        """
        Simulate given actions.

        Usage:
            - `simulate_action(state, action1)` will execute a single action, `action1`
            - `simulate_action(state, action1, action2)` will execute two consecutive actions, `action1` and `action2`
            - ...
            - `simulate_action(state, *action_list)` will execute actions in the order specified in the `action_list`

        :param state: State where the simulation starts from. If None, the simulation starts from the initial state.
        :param actions: Actions to simulate or execute.
        :param problem_type: Problem Type Index (Challenge problem #)
        :return: The last state after simulating all actions
        """
        if IS_DEBUG:  # Logging for debug
            self._logger.debug(f'------- SIMULATION START: {actions} -------')

        # Restore to the given state
        self.set_to_state(state)
        _prev_current = self.get_state()

        for act in actions:  # For each actions in the variable arguments,
            # Run actions through calling each action object. If error occurs, raise as it is (except for GameOver)
            try:
                act(self)
            except GameOver:
                break

            # Break the loop if the game ends within executing actions.
            if self.is_game_end():
                break

        self._current = self._save_state()

        if IS_DEBUG:  # Logging for debug
            self._logger.debug('State has been changed to: \n' + self._unique_game_state_identifier())
            self._logger.debug('\n' + print_board(self._board))
            self._logger.debug('------- SIMULATION ENDS -------')

        # Update memory usage
        self._update_memory_usage()

        return deepcopy(self._current)

    def _unique_game_state_identifier(self) -> str:
        """
        Return the unique identifier for game states.
        If two states are having the same identifier, then the states can be treated as identical in this problem.

        :return: String of game identifier
        """

        return self._board.partial_FEN()

    def _save_state(self) -> dict:
        """
        Helper function for saving the current state representation as a python dictionary from the game board.

        :return: State representation of a game (in basic python objects)
        """

        return {
            'state_id': self._unique_game_state_identifier(),
            # Unique identifier for the game state. If this is the same, then the state will be equivalent.
            'turn': self._board.turn,  # Currently playing Player's ID
            'board': {  # Information about the current board
                'fence_center': self._fences.copy()
            },
            'player': {
                p: {  # Information about the current player
                    'pawn': list(self._board.pawns[p].square.location),
                    'fences_left': self._board.fences_left[p]
                }
                for p in ['black', 'white']
            }
        }

    def _restore_state(self, state: dict):
        """
        Helper function to restore board state to given state representation.
        """
        for r,c in self._board.fence_center_grid.argwhere():
            self._board._place_or_remove_fence(r, c, 'h', place=False)
            self._board._place_or_remove_fence(r, c, 'v', place=False)
            
        # Clear everything
        self._board.fence_center_grid[:, :] = False
        self._board.horizontal_fence_grid[:, :] = False
        self._board.vertical_fence_grid[:, :] = False
        self._fences = []

        # Set players' current position
        for p in ['black', 'white']:
            self._board.pawns[p].move(self._board.get_square_or_none(*state['player'][p]['pawn']))

        # Recover route-related information (before building fences)
        for pawn in self._board.pawns.values():
            pawn.square.reset_neighbours()

        for pawn in self._board.pawns.values():
            self._board.update_neighbours(pawn.square)

        # Recover fences. Before recovery, give all the fences to the current player.
        self._board.fences_left['black'] = FENCES_MAX * 2
        self._board.fences_left['white'] = FENCES_MAX * 2

        # Re-simulate fencing:
        for (r, c), o in state['board']['fence_center']:
            current_player = self._board.current_player()
            if o == 'h':
                act = BLOCK(player=current_player, orientation='horizontal', edge=(r, c))
            elif o == 'v':
                act = BLOCK(player=current_player, orientation='vertical', edge=(r, c))
            else:
                raise ValueError(f'Fence center {r}, {c} is not in both horizontal and vertical fences')
            try:
                act(self, avoid_check=True)
            except GameOver:
                continue

        # Set players' information
        for p in ['black', 'white']:
            self._board.fences_left[p] = state['player'][p]['fences_left']

        self._board.turn = state['turn']


# Export only GameBoard and RESOURCES.
__all__ = ['GameBoard', 'IS_DEBUG', 'IS_RUN']
