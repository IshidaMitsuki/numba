# ga_train.py – GA 最適化（Numba 評価版）
from __future__ import annotations
import random, json, pathlib, time
import numpy as np
from deap import base, creator, tools

from core import FEATURES, DEFAULT_W        # 盤面定義とデフォルト重み
from numba_ga_train import evaluate_pop     # ★ Numba 一括評価

# ───────────────────────── パラメータ ─────────────────────────
POP, GEN = 8, 8
ELITE_SIZE = 2

# ───────────────────────── DEAP 準備 ─────────────────────────
creator.create('FitMax', base.Fitness, weights=(-1.0,))
creator.create('Ind', list, fitness=creator.FitMax)

tb = base.Toolbox()
tb.register('attr',   random.uniform, -1, 1)
tb.register('individual',
            tools.initRepeat, creator.Ind,
            tb.attr, n=len(FEATURES))
tb.register('population', tools.initRepeat, list, tb.individual)
tb.register('mate',   tools.cxBlend,     alpha=0.5)
tb.register('mutate', tools.mutGaussian, mu=0, sigma=0.3, indpb=0.3)
tb.register('select', tools.selTournament, tournsize=3)

# ─── Numba 評価ラッパー (個体1件→tuple) ───────────────────────
def evaluate(individual):
    arr = np.ascontiguousarray([individual], dtype=np.float64)
    score = float(evaluate_pop(arr)[0])
    return (score,)          # DEAP は tuple を要求

tb.register('evaluate', evaluate)

# ───────────────────────── main ──────────────────────────────
def main():
    # 1) 初期個体生成
    pop = tb.population(POP)

    # 1-a) best_weights.json → 個体0 へ注入
    best_path = pathlib.Path('best_weights.json')
    if best_path.exists():
        try:
            data = json.loads(best_path.read_text())
            if all(k in data for k in FEATURES):
                pop[0] = creator.Ind([data[f] for f in FEATURES])
                print('Seed individual loaded from best_weights.json')
        except Exception as e:
            print('Warning: best_weights.json load error ->', e)

    # 1-b) DEFAULT_W を個体1へ
    pop[1 % POP] = creator.Ind([DEFAULT_W[f] for f in FEATURES])
    print('Default-weight individual injected into generation 0')

    # 2) 進化ループ
    for g in range(GEN):
        # ── 評価 ─────────────────────────────
        fits = list(map(tb.evaluate, pop))
        for ind, fit in zip(pop, fits):
            ind.fitness.values = fit

        best_now = min(pop, key=lambda i: i.fitness.values[0])
        avg_now  = np.mean([i.fitness.values[0] for i in pop])
        print(f'Gen {g:02d}: best {best_now.fitness.values[0]:.1f}, '
              f'avg {-avg_now:.1f}  '
              f'w={[round(w, 3) for w in best_now]}')

        # ── 次世代生成 ───────────────────────
        elites   = tools.selBest(pop, ELITE_SIZE)
        selected = tb.select(pop, len(pop) - ELITE_SIZE)
        children = list(map(tb.clone, selected))

        # 交叉
        for c1, c2 in zip(children[::2], children[1::2]):
            if random.random() < 0.7:
                tb.mate(c1, c2)
                del c1.fitness.values, c2.fitness.values

        # 突然変異
        for c in children:
            if random.random() < 0.2:
                tb.mutate(c)
                del c.fitness.values

        pop = elites + children

    # 3) 最終世代の評価 & 保存
    fits = list(map(tb.evaluate, pop))
    for ind, fit in zip(pop, fits):
        ind.fitness.values = fit

    best = max(pop, key=lambda i: i.fitness.values[0])
    weights_dict = {f: w for f, w in zip(FEATURES, best)}
    pathlib.Path('best_weights.json').write_text(json.dumps(weights_dict))
    print('Saved → best_weights.json')


if __name__ == '__main__':
    start_all = time.time()
    main()
    print(f'Total GA training time: {time.time() - start_all:.1f} sec')
