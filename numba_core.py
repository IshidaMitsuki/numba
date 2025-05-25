# numba_core.py  – ビットボード + JIT AI
import numpy as np
from numba import njit, prange

MATRIX_W, MAX_ROW = 10, 40
from core import PIECE_SHAPES                         # 旧タプル定義を再利用
KIND2IDX = {k: i for i, k in enumerate("IOTSZJL")}

# ---------- SHAPES[int kind][int rot][4][dx,dy] ----------
SHAPES = np.zeros((7, 4, 4, 2), dtype=np.int64)
for k, kind in enumerate("IOTSZJL"):
    for r in range(4):
        for c, (dx, dy) in enumerate(PIECE_SHAPES[kind][r]):
            SHAPES[k, r, c, 0] = dx
            SHAPES[k, r, c, 1] = dy
# ----------------------------------------------------------

# ■ 高さ計算
@njit(cache=True)
def get_heights(cols):
    h = np.zeros(MATRIX_W, dtype=np.int64)
    for x in range(MATRIX_W):
        col, y = cols[x], 0
        while (col >> y) & 1:
            y += 1
        h[x] = y
    return h

# ■ 行消去（タッチ行のみ）
@njit(cache=True)
def clear_rows_fast(cols, touched):
    cleared = 0
    for r in touched:
        full = True
        for x in range(MATRIX_W):
            if ((cols[x] >> r) & 1) == 0:
                full = False
                break
        if not full:
            continue
        cleared += 1
        for x in range(MATRIX_W):
            lower = cols[x] & ((1 << r) - 1)
            upper = cols[x] >> (r + 1)
            cols[x] = lower | (upper << r)
    return cleared

# ■ ドロップ & 消去
@njit(cache=True)
def drop_piece(cols, kind, rot, x):
    shape = SHAPES[kind, rot]
    h     = get_heights(cols)
    ydrop = -1
    for i in range(4):
        dx, dy = shape[i, 0], shape[i, 1]
        cand   = h[x + dx] - dy
        if cand > ydrop:
            ydrop = cand
    touched = np.empty(4, dtype=np.int64)
    for i in range(4):
        dx, dy = shape[i, 0], shape[i, 1]
        ry = ydrop + dy
        cols[x + dx] |= 1 << ry
        touched[i] = ry
    return clear_rows_fast(cols, touched)

# ■ 特徴量（旧 _recalc と一致）
@njit(cache=True)
def calc_features(cols):
    heights = get_heights(cols)

    rough = bump = 0
    for x in range(MATRIX_W - 1):
        diff = abs(heights[x] - heights[x + 1])
        rough += diff
        if diff >= 2:
            bump += diff * diff

    holes = covered = 0
    for x in range(MATRIX_W):
        col, h = cols[x], heights[x]
        for y in range(h):
            if ((col >> y) & 1) == 0:
                holes += 1
                if h - y > 4:
                    covered += 1

    w_depth = w_cells = 0
    for x in range(MATRIX_W):
        l = heights[x - 1] if x > 0 else MAX_ROW
        r = heights[x + 1] if x < MATRIX_W - 1 else MAX_ROW
        if heights[x] < l and heights[x] < r:
            d = min(l, r) - heights[x]
            w_depth  = max(w_depth, d)
            w_cells += d

    # parity（行ごと XOR）
    p = 0
    for y in range(MAX_ROW):
        bits = 0
        for x in range(MATRIX_W):
            bits += (cols[x] >> y) & 1
        p ^= bits & 1

    return np.array([holes, covered, rough, bump,
                     w_depth, w_cells, p], dtype=np.float64)


# ==================================================================
# HeuristicAIJit  ―― 旧 Python 版 HeuristicAI と API 互換
# ==================================================================
@njit(cache=True)
def eval_position(cols, weights):
    feats = calc_features(cols)
    s = 0.0
    for i in range(weights.shape[0]):
        s += weights[i] * feats[i]
    return s

class HeuristicAIJit:
    """旧 HeuristicAI.best_move と同じ返り値 (rot, x) を返す"""
    def __init__(self, weights):
        self.w_arr = np.array([weights[f] for f in weights], dtype=np.float64)
        self.max_gain = float(np.max(self.w_arr))   # 枝刈り用と互換

    def best_move(self, board_cols, kind_chr, _next4=None):
        kind = KIND2IDX[kind_chr]
        best_s, best = -1e30, (0, 0)
        for r in range(4):
            for x in range(-2, MATRIX_W + 2):
                cols = board_cols.copy()
                try:
                    drop_piece(cols, kind, r, x)
                except:
                    continue
                s = eval_position(cols, self.w_arr)
                if s > best_s:
                    best_s, best = s, (r, x)
        return best

@njit(cache=True)
def best_move(cols, kind_idx, weights):
    best_score = -1e30
    best_r = best_x = 0

    for r in range(4):
        shape = SHAPES[kind_idx, r]
        for x in range(-2, MATRIX_W + 2):
            # --- 横はみ出しチェック (try/except は Numba 非対応) --------
            valid = True
            for i in range(4):
                dx = shape[i, 0]
                if not (0 <= x + dx < MATRIX_W):
                    valid = False
                    break
            if not valid:
                continue

            tmp = cols.copy()
            # drop_piece がはみ出しで失敗しない前提で呼ぶ
            cleared = drop_piece(tmp, kind_idx, r, x)

            score = eval_position(tmp, weights)
            # テトリス加点 (旧 HeuristicAI と同じ 10 点)
            if cleared == 4:
                score += 10.0

            if score > best_score:
                best_score = score
                best_r, best_x = r, x

    return best_r, best_x
