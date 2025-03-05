import traceback
from time import time
from logging import Logger
import psutil  # 프로세스 메모리때문에 추가함
import os

from pyquoridor.exceptions import InvalidMove, InvalidFence

from action import Action, MOVE
from board import GameBoard
from .util import Performance, MEGABYTES, load_ta_agent

HARD_TIME_LIMIT = 0
HARD_MEMORY_LIMIT = 0

def calculate_total_turns(board: GameBoard, solution: list[Action]) -> int:
    if not solution:
        return float('inf')
    
    total_turns = 0
    current_pos = None
    
    for action in solution:
        if isinstance(action, MOVE):
            if current_pos is None:
                player = action.player
                pawn_pos = board._board.pawns[player].square.location
                current_pos = (pawn_pos[0], pawn_pos[1])
            
            next_pos = action.position
            turns = board.get_move_turns(current_pos, next_pos)
            total_turns += turns
            current_pos = next_pos
            
    return total_turns

def execute_heuristic_search(agent, initial_state: dict, logger: Logger):
    # Build a dummy board
    board = GameBoard()
    board._initialize()

    # Load TA agent if exists
    agents = {'agent': agent}
    ta_agent = load_ta_agent(agent)
    if ta_agent is not None:
        agents['ta'] = ta_agent

    process = psutil.Process(os.getpid())
    
    # For each agent, execute the same problem.
    results = {}
    for k, a in agents.items():
        initial_memory = process.memory_info().rss / MEGABYTES
        
        # Initialize board and log initial memory size
        board.set_to_state(initial_state, is_initial=True)
        board.reset_memory_usage()
        board.get_current_memory_usage()

        solution = None
        failure = None
        peak_memory = initial_memory 

        # Start to search
        logger.info(f'Begin to search using {a.name} agent.')
        time_start = time()
        
        try:
            def update_memory():
                nonlocal peak_memory
                try:
                    current = process.memory_info().rss / MEGABYTES
                    peak_memory = max(peak_memory, current)
                except:
                    pass
            
            import threading
            stop_monitoring = False
            
            def memory_monitor():
                while not stop_monitoring:
                    update_memory()
                    import time
                    time.sleep(0.1)
            
            monitor_thread = threading.Thread(target=memory_monitor, daemon=True)
            monitor_thread.start()
            
            solution = a.heuristic_search(board)
            
            stop_monitoring = True
            monitor_thread.join(timeout=1.0)
            
            assert isinstance(solution, list), \
                'Solution should be a LIST of actions. The current outcome is not a list.'
            assert all(isinstance(s, Action) for s in solution), \
                'Solution should be a list of ACTIONs. It contains an element which is not an ACTION.'
        except:
            failure = traceback.format_exc()
        finally:
            stop_monitoring = True

        # Compute how much time passed -> 초 단위가 아닌 밀리초 단위로 측정
        time_end = time()
        time_delta = time_end - time_start  

        # board 내장 측정과 psutil 측정 중 더 큰 값 사용
        board_memory = board.get_max_memory_usage() / MEGABYTES
        memory_usage = max(board_memory, peak_memory - initial_memory)

        if k == 'agent' and time_delta > HARD_TIME_LIMIT > 0:
            return Performance(
                failure=f'Time limit exceeded! {time_delta:.3f} seconds passed!',
                outcome=float('inf'),
                search=None,
                time=time_delta,
                memory=memory_usage,
                point=1 # Just give submission point
            )
        if k == 'agent' and memory_usage > HARD_MEMORY_LIMIT > 0:
            return Performance(
                failure=f'Memory limit exceeded! {memory_usage:.2f} MB used!',
                outcome=float('inf'),
                search=None,
                time=time_delta,
                memory=memory_usage,
                point=1 # Just give submission point
            )

        is_end = False
        total_turns = float('inf')
        if solution is not None:
            try:
                board.set_to_state(initial_state, is_initial=True)  # Reset to initial state
                total_turns = calculate_total_turns(board, solution)
                board.simulate_action(None, *solution)
                is_end = board.is_game_end()
            except (InvalidMove, InvalidFence):
                failure = traceback.format_exc()

        results[k] = Performance(
            failure=failure,
            outcome=total_turns if solution is not None else float('inf'),
            search=None,
            time=time_delta,
            memory=memory_usage,
            point=1 + int(is_end)
        )

    # Give points by the stage where the agent is.
    res = results['agent']

    is_beating_ta_outcome = True
    is_beating_ta_time = False
    if 'ta' in results:
        is_beating_ta_outcome = ((results['ta'].outcome >= res.outcome > 0)
                                 or (results['ta'].outcome == res.outcome == 0))
        is_beating_ta_time = results['ta'].time >= res.time

    is_basic_stage = (res.failure is None) and (res.point > 1) and is_beating_ta_outcome
    is_intermediate_stage = is_basic_stage and (res.memory <= 10)
    is_advanced_stage = is_intermediate_stage and is_beating_ta_time

    return Performance(
        failure=res.failure,
        outcome=res.outcome,
        search=res.search,
        time=res.time,
        memory=res.memory,
        point=1 + int(is_basic_stage) + int(is_intermediate_stage) + int(is_advanced_stage) * 2
    )