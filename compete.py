# Parser for arguments
import logging
import random
from argparse import ArgumentParser
# A dictionary class which can set the default value
from collections import defaultdict
# Package for multiprocessing (evaluation will be done with multiprocessing)
from multiprocessing import Process, Queue
from queue import Empty
# Package for randomness and seed control
from random import seed, shuffle, randint
# Package for time management
from time import time, sleep
from traceback import format_exc
# Package for combination
from itertools import combinations

# Package for measuring process utilization (memory)
import psutil as pu

# Function for loading your agents
from agents.load import get_all_agents
from board import GameBoard, InvalidFence, InvalidMove
from action import Action

#: The number of games to run the evaluation
GAMES = 10
#: Limit of execution. (1 minute, 1GB per move)
TIME_LIMIT = 60
MEGABYTES = 1024 * 1024
MEMORY_LIMIT = 1024
#: Signals
QUIT_SIGNAL = 'QUIT'


def _query(player, player_side, query_queue: Queue, action_queue: Queue):
    try:
        from importlib import import_module
        module = import_module(f'agents.{player}')
        agent = module.Agent(player_side)
    except:
        action_queue.put(f'ERROR: Player {player} not found!')
        return

    board_for_player = GameBoard()
    board_for_player._initialize()
    while True:
        # Sleep 1 second to avoid blocking
        sleep(1)
        try:
            query = query_queue.get(timeout=TIME_LIMIT * 2)
        except Empty:
            continue

        if query == QUIT_SIGNAL:
            break

        board_for_player.set_to_state(query, is_initial=True)
        search_begin = time()

        try:
            action = agent.adversarial_search(board_for_player, time_limit=search_begin + TIME_LIMIT)
            action_queue.put(action)
        except:
            action_queue.put('ERROR!\n' + format_exc())


def _execute(player, player_side, query_queue: Queue, action_queue: Queue):
    """
    Execute a player algorithm.
    :param player: Agent
    :param player_side: Side of the player (either White or Black)
    :param query_queue: Queue for querying states
    :param action_queue: Queue for returning actions
    :return: A process
    """
    proc = Process(name=f'Player [{player}]',
                   target=_query, args=(player, player_side, query_queue, action_queue),
                   daemon=True)
    proc.start()
    return proc


def _info(msg):
    logging.info(msg)
    print(msg)


# Main function
if __name__ == '__main__':
    # Definition of arguments, used when running this program
    argparser = ArgumentParser()
    argparser.set_defaults(debug=False)
    argparser.add_argument('-p', '-players', '--players', type=str, nargs='+',
                           help='Players to compete (Program will evaluate your players as a league match)')
    argparser.add_argument('--debug', action='store_true', dest='debug',
                           help='Enable debug mode')
    args = argparser.parse_args()

    # Initialize logger
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format='%(asctime)s [%(name)-12s] %(levelname)-8s %(message)s',
                        filename='execution.log',
                        # Also, the output will be logged in 'execution.log' file.
                        filemode='w+',
                        force=True, encoding='UTF-8')  # The logging file will be overwritten.

    # Set a random seed
    seed(4242)
    summaries = []

    # Start evaluation process (using multi-processing)
    for players in combinations(args.players, 2):
        players = list(players)
        _info(f'Competing players: {players}')
        summaries.append(f'Competing players: {players}')
        winner_count = []

        for t in range(GAMES):
            if random.random() < 0.5:
                white, black = players
            else:
                black, white = players

            _info(f'Player {white} (white) vs {black} (black) in trial #{t + 1}')
            player_name = {
                'white': white,
                'black': black
            }
            query_queues = {
                side: Queue(2)
                for side in ['white', 'black']
            }
            action_queues = {
                side: Queue(2)
                for side in ['white', 'black']
            }

            # Start a board
            board = GameBoard()
            board._initialize()
            state = board.get_initial_state()
            processes = {
                side: _execute(player_name[side], side,
                               query_queues[side], action_queues[side])
                for side in ['white', 'black']
            }

            # Record memory usage
            start_mem = {}
            for side in ['white', 'black']:
                try:
                    start_mem[side] = pu.Process(processes[side].pid).memory_info().rss / MEGABYTES
                except pu.NoSuchProcess:
                    start_mem[side] = 0

            # Run the game (white first)
            turns = 0
            while not board.is_game_end():
                turns += 1
                side = board.current_player()
                other_side = 'white' if side == 'black' else 'black'
                state = board.get_state()
                query_queues[side].put(state)

                try:
                    # Give 5 seconds for set-up the board
                    result = action_queues[side].get(timeout=TIME_LIMIT + 5)
                    if type(result) is str:  # This indicates an error
                        winner_count.append(player_name[other_side])
                        _info(f'Error in player {side} [{player_name[side]}]: {result}')
                        break
                    if not isinstance(result, Action):
                        winner_count.append(player_name[other_side])
                        _info(f'Invalid action returned by player {side} [{player_name[side]}]: {result}')
                        break

                    # Check memory usage
                    try:
                        usage = pu.Process(processes[side].pid).memory_info().rss / MEGABYTES
                    except pu.NoSuchProcess:
                        usage = start_mem[side]

                    if usage - start_mem[side] > MEMORY_LIMIT:
                        # Memory overflow
                        winner_count.append(player_name[other_side])
                        _info(f'Memory Overflow in player {side} [{player_name[side]}]: '
                                     f'{usage - start_mem[side]}MB')
                        break

                    state = board.simulate_action(state, result)
                    # Reset state to check if the action is valid
                    board.set_to_state(state, is_initial=True)

                    _info(f'Player {side} [{player_name[side]}] do action {result}.')
                except Empty:
                    # Timeout happened
                    winner_count.append(player_name[other_side])
                    _info(f'Timeout of player {side} [{player_name[side]}]!')
                    break
                except (InvalidFence, InvalidMove):
                    # Invalid action
                    winner_count.append(player_name[other_side])
                    _info(f'Invalid action returned by player {side} [{player_name[side]}]!')
                    break
                except Exception:
                    # Invalid action
                    winner_count.append(player_name[other_side])
                    _info(f'Exception occurred by player {side} [{player_name[side]}]!')
                    _info(format_exc())
                    break
                
                if turns > 5000:
                    _info(f'Game is too long! Treat this game as a tie.')
                    break
            else:
                winner_count.append(player_name[side])
                _info(f'Game ended with the action of {side} [{player_name[side]}]!')

            # Clean up processes
            for side in ['white', 'black']:
                if processes[side].is_alive():
                    query_queues[side].put(QUIT_SIGNAL)
                    processes[side].join()
                else:
                    processes[side].terminate()

            msg = f'Winner of #{t+1} trial: {winner_count[-1]}'
            _info(msg)

        for player in players:
            win_count = sum(c == player for c in winner_count)
            winrate = win_count / len(winner_count) * 100

            msg = f'Winning rate of {player:10s}: {win_count:2d}/{GAMES:2d} = {winrate:6.2f}%'
            _info(msg)
            summaries.append(msg)

        _info('-' * 80)
        summaries.append('-' * 80)

    _info('=' * 80)
    _info('SUMMARY')
    _info('-' * 80)
    _info('\n'.join(summaries))
