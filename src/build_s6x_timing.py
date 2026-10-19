"""Build kaggle/rsna-knee-s6xtime/: the s6fall timing commit for C17 (candidates.md C17, handoff 2026-10-18 "Next action" 2).

C17 = C14 + the audit's fixes + s6fall's crop TTA (src/build_anchor_copy.py --anchor 09586s6fx).  This kernel runs C17's own s6fall
script (byte for byte the text C17 writes) as a module and drives its B stage (heliosli's compact_infer through nartaa's launcher) on
training studies, two GPU processes at once as on the hidden test:

  phase 1 (timing)  ~520 non-gold training studies, two cost-balanced halves at the same time: GPU 0 plain (K1_CROPS empty = C14's
                    path), GPU 1 with C17's crops.  Steady-state s/study of each -> the projected hidden-test time (1,284 studies).
  phase 2 (checks)  the 58 gold studies twice: GPU 0 plain, GPU 1 with 9 crops (centre, 4 corners, 4 mid-diagonals) dumped per crop.
                    The patched path's centre crop must equal the plain path; macro AUC of centre / 3 / 5-corner / 5-mid / 9 crops is a
                    direction read only (58 studies).

The study list is drawn inside the kernel (fixed seed) from train.csv: no StudyInstanceUID is written to this repo (public, traps
73) or to the kernel's output; the output holds aggregates only (s6x_timing.json).

    export PYTHONUTF8=1
    .venv/Scripts/python.exe src/build_s6x_timing.py [--check]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_anchor_copy import ANCHORS, s6fall_cells  # noqa: E402
from build_fork import dump_json  # noqa: E402
from build_fork954 import DOCKER_IMAGE, S6FALL_DATASET, check_no_competition_ids, module_level_names  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLUG = "rsna-knee-s6xtime"
N_POOL = 520
SEED = 20261019
CROPS9 = ((16, 16), (0, 0), (0, 32), (32, 0), (32, 32), (8, 8), (8, 24), (24, 8), (24, 24))

INTRO = """# s6fall timing commit for C17

Built by `src/build_s6x_timing.py`. The next cell writes C17's s6fall script (nartaa's s6fall as C14 runs it, plus C17's patches:
crop TTA, notebook-time guard, partial results). The driver runs its B stage on training studies drawn here with a fixed seed:
phase 1 times plain vs C17's crops on two cost-balanced halves at once (one per GPU); phase 2 scores the 58 gold studies plain and
with 9 crops (per-crop dump) to check the patched centre crop against the plain path. Output: aggregates only (`s6x_timing.json`).
"""

DRIVER = r'''# s6fall timing commit for C17 (src/build_s6x_timing.py): C17's s6fall script as a module, its B stage on training studies
import importlib.util, json, os, subprocess, sys, time
import numpy as np
import pandas as pd
T0 = time.time()
N_POOL, SEED, CROPS9 = __N_POOL__, __SEED__, "__CROPS9__"
ROOT = '/tmp/s6xroot'
os.makedirs(ROOT + '/working', exist_ok=True)
if not os.path.lexists(ROOT + '/input'):
    os.symlink('/kaggle/input', ROOT + '/input')
os.environ['K1_KAGGLE_ROOT'] = ROOT
_spec = importlib.util.spec_from_file_location('s6f_c17', '/tmp/s6fall_run.py')
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)
CROPS_C17 = os.environ['K1_CROPS']                     # the script's own default (setdefault at import) = what C17 runs
S.OUT.mkdir(parents=True, exist_ok=True)
S.TMP.mkdir(parents=True, exist_ok=True)
S._RECEIPT['env'] = S.probe_env()
print('ENV', json.dumps(S._RECEIPT['env']), flush=True)
I = S.prepare_inputs()
print('BPROBE', json.dumps(S.probe_b_stack(I)), flush=True)   # imports the launcher with C17's crops: the main() patch applies
SUMMARY = {'crops_c17': CROPS_C17, 'n_pool': N_POOL, 'seed': SEED, 'c17_receipt': S._RECEIPT.get('c17')}

tr = pd.read_csv(I.comp / 'train.csv', dtype={'StudyInstanceUID': str})
gold_mask = tr[S.LAB].notna().all(axis=1)
gold = tr.loc[gold_mask, 'StudyInstanceUID'].tolist()
pool = sorted(tr.loc[~gold_mask, 'StudyInstanceUID'].tolist())
pick = [str(u) for u in np.random.default_rng(SEED).choice(pool, N_POOL, replace=False)]
ser = pd.read_csv(I.comp / 'train_series.csv', dtype=str, usecols=['StudyInstanceUID'])
COST = ser['StudyInstanceUID'].value_counts().to_dict()
SUMMARY.update(n_gold=len(gold), n_series_pool_sample=int(sum(COST.get(u, 0) for u in pick)))
print('studies: gold %d, pool sample %d of %d' % (len(gold), len(pick), len(pool)), flush=True)


def launch(gi, ids, crops, tag, dump=False):
    pdir = S.OUT / tag
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / 'ids.json').write_text(json.dumps(ids))
    cmd = S.b_cmd(I, 'train', pdir / 'ids.json', pdir, S.TMP / (tag + '_tmp'), time.time() + 4 * 3600)
    env = S.b_env(gi)
    env['K1_CROPS'] = crops
    env.pop('K1_CROP_DUMP', None)
    if dump:
        (pdir / 'crops').mkdir(exist_ok=True)
        env['K1_CROP_DUMP'] = str(pdir / 'crops')
    fh = open(pdir / 'log.txt', 'w')
    p = subprocess.Popen(cmd, env=env, cwd=str(S.TMP), stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         start_new_session=True)
    return {'tag': tag, 'p': p, 'fh': fh, 'pdir': pdir, 'n': len(ids), 'crops': crops, 't0': time.time()}


def wait_all(runs, cap_s):
    t_end = time.time() + cap_s
    last = 0.0
    while any(r['p'].poll() is None for r in runs) and time.time() < t_end:
        time.sleep(15)
        if time.time() - last > 300:
            last = time.time()
            for r in runs:
                prog = S.progress_lines(r['pdir'] / 'log.txt')
                print('[%5.2fh] %s rc=%s %s' % ((time.time() - T0) / 3600, r['tag'], r['p'].poll(),
                                               json.dumps(prog[-1]) if prog else S.tail(r['pdir'] / 'log.txt', 300)[-300:]), flush=True)
    for r in runs:
        if r['p'].poll() is None:
            S.kill_group(r['p'].pid)
        r['p'].wait()
        r['fh'].close()
        r['seconds'] = round(time.time() - r['t0'], 1)


def read_run(r):
    out = {'tag': r['tag'], 'crops': r['crops'], 'n': r['n'], 'rc': r['p'].returncode, 'wall_s': r['seconds']}
    log = S.tail(r['pdir'] / 'log.txt', 10 ** 7)
    for l in log.splitlines():
        if l.startswith('MODEL_READY'):
            out['model_ready'] = json.loads(l[len('MODEL_READY '):])
    try:
        rec = json.loads((r['pdir'] / 'receipt.json').read_text())
        out.update(scored=rec['scored'], fallback=len(rec['fallback']), timed_out=rec['timed_out'], seconds=rec['seconds'],
                   stats=rec['stats'])
    except Exception as e:
        out['receipt_error'] = repr(e)[:300]
        print('LOG TAIL', r['tag'], log[-3000:], flush=True)
    prog = S.progress_lines(r['pdir'] / 'log.txt')
    if prog:
        a = next((x for x in prog if x['done'] >= 20), prog[0])
        b = prog[-1]
        if b['done'] > a['done']:
            out['steady_s_per_study'] = round((b['elapsed_s'] - a['elapsed_s']) / (b['done'] - a['done']), 3)
        out['progress_last'] = b
    st = out.get('stats') or {}
    if st.get('rows'):
        out['per_study'] = {k: round(st[k] / st['rows'], 3) for k in ('decode_s', 'render_s', 'anchor_s', 'gpu_s', 'wait_s')}
        out['windows_per_study'] = round(st['windows'] / st['rows'], 1)
    return out


def safe_read(r):
    try:
        return read_run(r)
    except Exception as e:
        return {'tag': r['tag'], 'read_error': repr(e)[:300]}


def save():
    with open('/kaggle/working/s6x_timing.json', 'w') as fh_:
        json.dump(SUMMARY, fh_, indent=1)


def preds(r):
    pc = pd.read_csv(r['pdir'] / 'predictions.csv', dtype={'StudyInstanceUID': str})
    return pc.set_index('StudyInstanceUID')[S.LAB]


def crop_dump(r):
    recs = {}
    for f in sorted((r['pdir'] / 'crops').glob('crops_*.jsonl')):
        for line in f.read_text().splitlines():
            d = json.loads(line)
            recs[d['uid']] = (d['crops'], np.asarray(d['p'], np.float64))
    return recs


# ---- phase 1: timing, plain (GPU 0) vs C17's crops (GPU 1), two cost-balanced halves at once ----
parts = S.split_parts(pick, COST, 2)
SUMMARY['phase1_series'] = [int(sum(COST.get(u, 0) for u in p)) for p in parts]
runs = [launch(0, parts[0], '', 'p1_plain'), launch(1, parts[1], CROPS_C17, 'p1_c17', dump=True)]
wait_all(runs, 3.0 * 3600)
SUMMARY['phase1'] = [safe_read(r) for r in runs]
print('PHASE1', json.dumps(SUMMARY['phase1']), flush=True)
save()
try:
    dmp = crop_dump(runs[1])
    rho_c, rho_m = [], []
    P = np.stack([v[1] for v in dmp.values()])             # (studies, crops, 12)
    for j in range(len(S.LAB)):
        cen = pd.Series(P[:, 0, j])
        rho_c.append([round(float(cen.corr(pd.Series(P[:, k, j]), method='spearman')), 4) for k in range(1, P.shape[1])])
        rho_m.append(round(float(cen.corr(pd.Series(P[:, :, j].mean(1)), method='spearman')), 4))
    SUMMARY['phase1_crop_rho'] = {'studies': int(P.shape[0]), 'centre_vs_each_crop_min': float(np.min(rho_c)),
                                  'centre_vs_each_crop_mean': float(np.mean(rho_c)), 'centre_vs_mean_per_finding': dict(zip(S.LAB, rho_m))}
except Exception as e:
    SUMMARY['phase1_crop_rho'] = {'error': repr(e)[:300]}

# ---- phase 2: the 58 gold studies, plain (GPU 0) and 9 crops dumped (GPU 1) ----
runs2 = [launch(0, gold, '', 'p2_plain'), launch(1, gold, CROPS9, 'p2_crops9', dump=True)]
wait_all(runs2, 1.0 * 3600)
SUMMARY['phase2'] = [safe_read(r) for r in runs2]
save()
try:
    from sklearn.metrics import roc_auc_score
    y = tr.set_index('StudyInstanceUID').loc[gold, S.LAB].astype(int)
    plain = preds(runs2[0])
    dmp = crop_dump(runs2[1])
    ids = [u for u in gold if u in dmp and u in plain.index]
    crops = [tuple(c) for c in dmp[ids[0]][0]]
    P = np.stack([dmp[u][1] for u in ids])                 # (studies, 9, 12), crop order = CROPS9
    cen = P[:, crops.index((16, 16)), :]
    SUMMARY['phase2_centre_vs_plain_max_abs'] = float(np.abs(cen - plain.loc[ids].to_numpy(np.float64)).max())
    sets = {'centre': [(16, 16)], 'c3_diag': [(16, 16), (0, 0), (32, 32)],
            'c5_corners': [(16, 16), (0, 0), (0, 32), (32, 0), (32, 32)],
            'c5_mid': [(16, 16), (8, 8), (8, 24), (24, 8), (24, 24)], 'c9': crops}
    yy = y.loc[ids]
    auc = {}
    for nm, cs in sets.items():
        pm = P[:, [crops.index(tuple(c)) for c in cs], :].mean(1)
        per = [roc_auc_score(yy[l].values, pm[:, j]) for j, l in enumerate(S.LAB) if 0 < yy[l].sum() < len(yy)]
        auc[nm] = round(float(np.mean(per)), 4)
    SUMMARY['phase2_gold_macro_auc'] = auc
    SUMMARY['phase2_gold_n'] = len(ids)
except Exception as e:
    SUMMARY['phase2_error'] = repr(e)[:500]

# ---- projection to the hidden test (1,284 studies over two GPU processes) ----
proj = {}
for r in SUMMARY['phase1']:
    if r.get('steady_s_per_study'):
        proj[r['tag']] = round((642 * r['steady_s_per_study'] + (r.get('model_ready') or {}).get('seconds', 0)) / 3600, 3)
SUMMARY['projected_hidden_hours_per_process'] = proj
if len(proj) == 2:
    SUMMARY['projected_extra_hours_c17'] = round(proj['p1_c17'] - proj['p1_plain'], 3)
SUMMARY['total_hours'] = round((time.time() - T0) / 3600, 3)
save()
print('TIMING_SUMMARY', json.dumps(SUMMARY), flush=True)
'''


def build():
    cells, stage_src = s6fall_cells(ANCHORS["09586s6fx"]["s6fall"], "09586s6fx")
    script_cell = cells[1]
    assert script_cell["source"].startswith("%%writefile /tmp/s6fall_run.py\n")
    drv = (DRIVER.replace("__N_POOL__", str(N_POOL)).replace("__SEED__", str(SEED))
           .replace("__CROPS9__", ";".join("%d,%d" % c for c in CROPS9)))
    compile(drv, "s6xtime-driver", "exec")
    module_level_names(drv)

    def code(cid, src):
        return {"id": cid, "cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": src}
    nb = {"cells": [{"id": "s6xtime-intro", "cell_type": "markdown", "metadata": {}, "source": INTRO},
                    code("s6xtime-script", script_cell["source"]), code("s6xtime-driver", drv)],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"},
                       "rsna_s6x_timing": {"builder": "src/build_s6x_timing.py", "c17_anchor": "09586s6fx", "n_pool": N_POOL,
                                           "seed": SEED}},
          "nbformat": 4, "nbformat_minor": 4}
    meta = {"id": "tiankljucanin/" + SLUG, "title": "RSNA Knee S6xtime", "code_file": SLUG + ".ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
            "keywords": [], "dataset_sources": [S6FALL_DATASET, "heliosli/rsna-orthofoundation-fold0-epoch7-20261004"],
            "kernel_sources": [], "competition_sources": ["rsna-knee-abnormality-detection"], "model_sources": [],
            "docker_image": DOCKER_IMAGE, "machine_shape": "NvidiaTeslaT4"}
    return nb, meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    os.chdir(ROOT)
    nb, meta = build()
    nb_text, meta_text = dump_json(nb), dump_json(meta)
    check_no_competition_ids(nb_text, "the rendered notebook")
    out_dir = os.path.join("kaggle", SLUG)
    paths = ((os.path.join(out_dir, SLUG + ".ipynb"), nb_text), (os.path.join(out_dir, "kernel-metadata.json"), meta_text))
    if a.check:
        ok = all(os.path.exists(p) and open(p, encoding="utf-8").read() == t for p, t in paths)
        print("check: " + ("ok" if ok else "FAILED"))
        return 0 if ok else 1
    os.makedirs(out_dir, exist_ok=True)
    for p, t in paths:
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            f.write(t)
    print("wrote %s: %d cells" % (paths[0][0], len(nb["cells"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
