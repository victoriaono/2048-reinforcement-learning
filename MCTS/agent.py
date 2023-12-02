from game import Game2048
import random, copy
import sys
import time
from datetime import timedelta

NUM_ITERS = 100       
NUM_TRIALS = 100
EVAL_METHOD = 0

def random_run(game):
    game_copy = copy.deepcopy(game)
    while not game_copy.game_end: 
        move = random.randint(0, 3)
        game_copy.make_move(move)
    
    return game_copy.get_sum()

def monte_carlo_iter(game):
    total_score = [0, 0, 0, 0]

    # For each move (0 - 3)
    for move in range(0,4):
        game_copy = copy.deepcopy(game)
        game_copy.make_move(move)
        if str(game_copy) == str(game):
            continue

        # Try lots of paths with that move using random rollout policy
        for _ in range(NUM_ITERS):
            total_score[move] += random_run(game_copy)

    best_move = total_score.index(max(total_score))
    game.make_move(best_move)
    print(game)

def monte_carlo_run():
    game = Game2048()
    i = 0
    while not game.game_end:
        print("Iteration: ", i)
        monte_carlo_iter(game)
        i += 1

    print("Max Square Value: {}".format(game.max_num()))
    print("Total Square Sum: {}".format(game.get_sum()))
    print("Total Merge Score: {}".format(game.get_merge_score()))
    return game.max_num(), game.get_sum(), game.get_merge_score()

def main():
    if len(sys.argv) != 3:
        raise Exception("usage: python3 agent.py <num_iters> <num_trials>")

    global NUM_ITERS 
    global NUM_TRIALS 
    global EVAL_METHOD
    NUM_ITERS = int(sys.argv[1])
    NUM_TRIALS = int(sys.argv[2])
    EVAL_METHOD = 0 # Eval method 0: sum of all tiles
    print(f"starting MCTS with {NUM_ITERS} rollouts and {NUM_TRIALS} trials")
    
    max_val_results = [0] * NUM_TRIALS
    total_sum_results = [0] * NUM_TRIALS
    total_merge_score = [0] * NUM_TRIALS
    
    start_time = time.time()
    for i in range(NUM_TRIALS):
        max_val_results[i], total_sum_results[i], total_merge_score[i] = monte_carlo_run()
    end_time = time.time()
        
    total_sum_avg = sum(total_sum_results) / NUM_TRIALS
    max_val_avg = sum(max_val_results) / NUM_TRIALS
    total_merge_avg = sum(total_merge_score) / NUM_TRIALS

    f = open("monte_carlo_" + str(NUM_ITERS) + "_" + str(NUM_TRIALS) + "_" + str(EVAL_METHOD) + ".txt", "w")
    f.write("avg max val: " + str(max_val_avg) + "\n") 
    f.write("avg total sum: " + str(total_sum_avg) + "\n")
    f.write("avg merge score: " + str(total_merge_avg) + "\n")
    f.write("max vals: " + str(max_val_results) + "\n") 
    f.write("total sums: " + str(total_sum_results) + "\n")
    f.write("total merge score: " + str(total_merge_score) + "\n")
    f.close()

    print("total sum avg: " + str(total_sum_avg))
    print("max val avg: " + str(max_val_avg))
    print("merge score avg: " + str(total_merge_avg))
    print()
    print("time taken: ", str(timedelta(seconds=(end_time - start_time))))

if __name__ == '__main__':
    main()