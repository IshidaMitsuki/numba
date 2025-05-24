# numba_core.py
import numpy as np
from numba import njit, prange, int64, float64

MATRIX_W = 10                # 列数
MAX_ROW  = 40                # ビット長（0-39 を使用）

# ── ピース形状 (dx,dy) ⟂ CPython 用 tuple から int64 配列に変換 ──
from core import PIECE_SHAPES          # 既存定義を再利用
SHAPES = np.zeros((7, 4, 4, 2), dtype=np.int64)
KIND2IDX = {'I': 0, 'O': 1, 'T': 2, 'S': 3, 'Z': 4, 'J': 5, 'L': 6}
for k, kind in enumerate("IOTSZJL"):
    for r in range(4):
        for c, (dx, dy) in enumerate(PIECE_SHAPES[kind][r]):
            SHAPES[k, r, c, 0] = dx
            SHAPES[k, r, c, 1] = dy

# ───────────────── 高速ユーティリティ ─────────────────
@njit(cache=True)
def get_heights(cols):
    """各列の高さ (= 初めて空セルが現れる行番号)"""
    h = np.zeros(MATRIX_W, dtype=int64)
    for i in range(MATRIX_W):
        col = cols[i]
        y = 0
        while (col >> y) & 1:
            y += 1
        h[i] = y
    return h

@njit(cache=True)
def clear_rows_fast(cols, touched):
    """touched 行に限定してライン消去を行い、消去数を返す"""
    cleared = 0
    for ti in range(touched.shape[0]):
        r = touched[ti]
        if r < 0:
            continue
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

@njit(cache=True)
def drop_piece(cols, kind_idx, rot, x):
    """ビットボードにピースを落として行消去し、消去数を返す"""
    shape = SHAPES[kind_idx, rot]
    heights = get_heights(cols)
    y_drop = -1
    # 最低 y を決める
    for j in range(4):
        dx, dy = shape[j, 0], shape[j, 1]
        col_h = heights[x + dx] - dy
        if col_h > y_drop:
            y_drop = col_h
    # 配置
    touched = np.empty(4, dtype=int64)
    for j in range(4):
        dx, dy = shape[j, 0], shape[j, 1]
        ry = y_drop + dy
        cols[x + dx] |= 1 << ry
        touched[j] = ry
    # 消去
    cleared = clear_rows_fast(cols, touched)
    return cleared

# ───────────────── 特徴量計算 ─────────────────
@njit(cache=True)
def calc_features(cols):
    heights = get_heights(cols)

    # roughness / bump_pen
    rough = 0
    bump_pen = 0
    for i in range(MATRIX_W - 1):
        diff = abs(heights[i] - heights[i + 1])
        rough += diff
        if diff >= 2:
            bump_pen += diff * diff

    # holes / covered_holes
    holes = 0
    covered = 0
    for x in range(MATRIX_W):
        col = cols[x]
        h = heights[x]
        for y in range(h):  # 0 .. h-1
            if ((col >> y) & 1) == 0:          # 空
                holes += 1
                if h - y > 4:                  # 例: 4セルより深い穴
                    covered += 1

    # well depth & cells
    well_depth = 0
    well_cells = 0
    for x in range(MATRIX_W):
        left = heights[x - 1] if x > 0 else MAX_ROW
        right = heights[x + 1] if x < MATRIX_W - 1 else MAX_ROW
        if heights[x] < left and heights[x] < right:
            wd = min(left, right) - heights[x]
            well_depth = max(well_depth, wd)
            well_cells += wd

    # parity (偶奇段差)
    parity = sum((heights[i] & 1) for i in range(MATRIX_W)) & 1

    return np.array([holes, covered, rough, bump_pen,
                     well_depth, well_cells, parity], dtype=np.float64)

# ───────────────── ベストムーブ探索 ─────────────────
@njit(cache=True)
def best_move(cols, kind_idx, weights):
    best_score = -1e20
    best_r = 0
    best_x = 0
    for r in range(4):
        for x in range(-2, MATRIX_W + 2):
            # deepcopy
            tmp = cols.copy()
            # try
            try:
                cleared = drop_piece(tmp, kind_idx, r, x)
            except:
                continue
            feats = calc_features(tmp)
            score = 0.0
            for i in range(weights.shape[0]):
                score += weights[i] * feats[i]
            # テトリス等を加点する場合は score += cleared == 4 ...
            if score > best_score:
                best_score = score
                best_r = r
                best_x = x
    return best_r, best_x
