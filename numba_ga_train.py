# numba_ga_train.py
import numpy as np
from numba import njit, prange
from numba_core import drop_piece, best_move, SHAPES, calc_features, MATRIX_W

# GA 係数
MAX_PIECES   = 3000
LINE_TARGET  = 40
TOP_PENALTY  = 2000
FEATURE_COUNT = 7            # calc_features が返す長さ

@njit(cache=True)
def lcg(state):
    return (state * 1664525 + 1013904223) & 0xFFFFFFFF

@njit(cache=True)
def run_game(weights, seed):
    cols = np.zeros(MATRIX_W, dtype=np.int64)
    rnd  = seed
    pieces = 0
    lines  = 0
    bag = np.arange(7, dtype=np.int64)  # 0..6

    while pieces < MAX_PIECES and lines < LINE_TARGET:
        # bag pop
        if pieces % 7 == 0:
            # Fisher–Yates shuffle
            for i in range(6, 0, -1):
                rnd = lcg(rnd)
                j = rnd % (i + 1)
                bag[i], bag[j] = bag[j], bag[i]
        kind = bag[pieces % 7]

        # 最善手
        r, x = best_move(cols.copy(), kind, weights)

        # ドロップ
        cleared = drop_piece(cols, kind, r, x)
        lines += cleared
        pieces += 1

    score = -pieces
    if lines < LINE_TARGET:
        score -= TOP_PENALTY
    return score

@njit(parallel=True, cache=True)
def evaluate_pop(pop_weights):
    n = pop_weights.shape[0]
    fits = np.empty(n, dtype=np.float64)
    for i in prange(n):
        w = pop_weights[i]
        fits[i] = (run_game(w, 0) + run_game(w, 1) + run_game(w, 2)) / 3.0
    return fits

# テスト
if __name__ == "__main__":
    POP = 8
    pop = np.random.uniform(-1, 1, (POP, FEATURE_COUNT))
    fit = evaluate_pop(pop)
    print("fitness:", fit)
