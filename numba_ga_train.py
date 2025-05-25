# numba_ga_train.py – GA 用シミュレータ (旧 run_game と同じ評価)
import numpy as np
from numba import njit, prange
from numba_core import drop_piece, best_move, MATRIX_W

MAX_PIECES   = 3000
LINE_TARGET  = 40
TOP_PENALTY  = 2000

@njit(cache=True)
def lcg(x):
    return (x * 1664525 + 1013904223) & 0xFFFFFFFF

@njit(cache=True)
def run_game(weights, seed):
    cols   = np.zeros(MATRIX_W, dtype=np.int64)
    rnd    = seed
    pieces = lines = tetris = 0
    bag    = np.arange(7, dtype=np.int64)

    while pieces < MAX_PIECES and lines < LINE_TARGET:
        if pieces % 7 == 0:                      # 7-bag シャッフル
            for i in range(6, 0, -1):
                rnd = lcg(rnd)
                j   = rnd % (i + 1)
                bag[i], bag[j] = bag[j], bag[i]
        kind = bag[pieces % 7]

        r, x = best_move(cols.copy(), kind, weights)
        cleared = drop_piece(cols, kind, r, x)

        if cleared == 4:
            tetris += 1
        lines  += cleared
        pieces += 1

    score = -pieces + tetris * 10               # 旧版と同じ
    if lines < LINE_TARGET:
        score -= TOP_PENALTY
    return score

@njit(parallel=True, cache=True)
def evaluate_pop(pop_w_arr):
    n = pop_w_arr.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in prange(n):
        w = pop_w_arr[i]
        out[i] = (run_game(w, 0) +
                  run_game(w, 1) +
                  run_game(w, 2)) / 3.0
    return out
