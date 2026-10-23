"""Build the solution write-up: TikZ / pgfplots figures -> PNG + PDF, theme-aware inline SVG -> HTML, HTML -> PDF.

Inputs (all in docs/writeup/src/): style.tex + pgfstyle.tex (the shared figure style: Latin Modern, one block
vocabulary, token colours), fig_*.tex, chartdata/*.dat (chart data), template.html (the page; {{fig:name|alt}} and
{{eq:name|alt}} mark where figures and equations go). The display equations are the EQ dict below.

Outputs:
  figures/<name>.png, figures/<name>.pdf   every figure and equation, 300 dpi PNG on white + vector PDF
  rsna-knee-solution.html                  standalone page (fonts from fonts/, follows the system light / dark theme)
  rsna-knee-solution.pdf                   A4, light theme, printed by headless Edge / Chrome
  build/artifact.html                      the same page as an Artifact fragment (no doctype; publish with fonts/)

Tools: Tectonic (tools/tectonic.exe, $TECTONIC, or on PATH; fetches TeX packages on first use), pdftocairo (Poppler),
and Edge or Chrome for the PDF. Run from the repo root:
  .venv/Scripts/python.exe docs/writeup/build.py           # figures + HTML + PDF
  .venv/Scripts/python.exe docs/writeup/build.py --data    # first regenerate src/chartdata/ from artifacts/lb_final/
The numbers in the text are written in src/template.html; --data refreshes only the charts.
Fonts: Latin Modern (GUST Font License), subset to WOFF2.
"""
import argparse, glob, os, re, shutil, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SRC = os.path.join(HERE, 'src')
BUILD = os.path.join(HERE, 'build')
FIGS = os.path.join(HERE, 'figures')
HTML = os.path.join(HERE, 'rsna-knee-solution.html')
PDF = os.path.join(HERE, 'rsna-knee-solution.pdf')
FRAGMENT = os.path.join(BUILD, 'artifact.html')

K_FIG = 1.55   # px per pt for figures (10 pt figure text -> 15.5 px)
K_EQ = 1.80    # px per pt for display equations (10 pt math -> 18 px, the body size)
PNG_DPI = 300

EQ = {
    'eq_auc': r'\mathrm{AUC}_{\mathrm{macro}}=\frac{1}{12}\sum_{k=1}^{12}\mathrm{AUC}\bigl(\{\hat y_{ik}\}_i,\{y_{ik}\}_i\bigr)',
    'eq_target': r'q_{ik}=\hat F^{-1}_{\bar p,k}\!\left(\frac{\operatorname{rank}_k(r_{ik})-\frac12}{n}\right),\qquad y^{\mathrm t}_{ik}=(1-\lambda)\,\bar p_{ik}+\lambda\,q_{ik},\quad \lambda=0.5',
    'eq_conf': r'c_{ik}=\operatorname{clip}\Bigl(\tfrac12\bigl(1-2\,\sigma_{ik}\bigr)+\tfrac12\bigl|2\bar p_{ik}-1\bigr|,\;0.15,\;1\Bigr)',
    'eq_head': r"a_{\ell w}=\frac{m_w\exp g_\ell(\tilde h_w)}{\sum_{w'}m_{w'}\exp g_\ell(\tilde h_{w'})},\qquad z_\ell=\Bigl\langle\sum_w a_{\ell w}\,\tilde h_w,\;v_\ell\Bigr\rangle+b_\ell",
    'eq_loss': r'\mathcal L=\frac1B\sum_{i=1}^{B}\frac{\sum_k c_{ik}\,\mathrm{BCE}\bigl(z_{ik},\,y^{\mathrm t}_{ik}\bigr)}{\sum_k c_{ik}}',
    'eq_rank': r's_{ik}=\sum_m \omega_{mk}\,\pi^{(m)}_{ik},\qquad \pi^{(m)}_{ik}=\frac{\operatorname{rank}_k\bigl(\hat y^{(m)}_{ik}\bigr)}{N},\qquad \sum_m\omega_{mk}=1',
    'eq_fork': r'\mathrm{LB}(\beta)\approx(1-\beta)\,\mathrm{LB}_{\mathrm{anchor}}+\beta\,\mathrm{LB}_{\mathrm{leg}}+\kappa\,\beta(1-\beta)',
}

TOKENS = {  # TeX colour (RGB 0-255) -> CSS token suffix; style.tex defines the colours, template.html the tokens
    (0, 0, 0): 'ink', (90, 90, 90): 'inkb', (200, 200, 200): 'grid', (191, 191, 191): 'grid', (255, 255, 255): 'paper',
    (231, 238, 246): 'data', (246, 238, 224): 'model', (232, 243, 234): 'target', (248, 233, 226): 'pub',
    (42, 120, 214): 'own', (235, 104, 52): 'pubs',
}
RGB = re.compile(r'rgb\(\s*([\d.]+)%\s*,\s*([\d.]+)%\s*,\s*([\d.]+)%\s*\)')


def tool(name, env, candidates):
    for c in [os.environ.get(env, '')] + candidates + [shutil.which(name) or '']:
        if c and os.path.exists(c):
            return c
    raise SystemExit(f'{name} not found (set ${env})')


# ---------------------------------------------------------------- chart data
def make_data():
    import pandas as pd
    lb = os.path.join(REPO, 'artifacts', 'lb_final')
    t = pd.read_csv(os.path.join(lb, 'our_entries_pub_priv.csv'))
    public = t.desc.str.match(r'^(C[0-9]|P-27|anchor-only|Hedge|P-40|A14|C21)')   # entries with public members
    t['grp'] = ['pub' if p else 'own' for p in public]
    t.loc[t.no == 28, 'grp'] = 'probe'                                              # the deliberate per-finding probe
    d = os.path.join(SRC, 'chartdata')   # not "data/": .gitignore keeps every data/ folder out of git
    s = t[(t.pub >= 0.90) & (t.grp != 'probe')]
    p = t[(t.pub >= 0.82) & (t.grp != 'probe') & (t.no != 1)]
    for g in ('own', 'pub'):
        s[s.grp == g][['no', 'pub', 'priv']].to_csv(os.path.join(d, f'entries_{g}.dat'), sep=' ', index=False)
        p[p.grp == g][['no', 'pub', 'priv']].to_csv(os.path.join(d, f'prog_{g}.dat'), sep=' ', index=False)
    pv = pd.read_csv(os.path.join(lb, 'private_lb.csv'))
    h = pv.score.round(3).value_counts().sort_index()
    h = h[(h.index >= 0.915) & (h.index <= 0.961)]
    h.rename_axis('score').reset_index(name='n').to_csv(os.path.join(d, 'field_hist.dat'), sep=' ', index=False)
    print(f'data: {len(s)} entries >= 0.90 public ({(s.grp == "own").sum()} own), {len(pv)} teams')


# ---------------------------------------------------------------- TeX -> PDF / PNG
def compile_tex():
    tectonic = tool('tectonic', 'TECTONIC', [os.path.join(REPO, 'tools', 'tectonic.exe'), os.path.join(REPO, 'tools', 'tectonic')])
    os.makedirs(BUILD, exist_ok=True)
    os.makedirs(FIGS, exist_ok=True)
    for f in glob.glob(os.path.join(SRC, '*.tex')) + glob.glob(os.path.join(SRC, 'chartdata', '*.dat')):
        shutil.copy(f, BUILD)
    for name, body in EQ.items():
        with open(os.path.join(BUILD, name + '.tex'), 'w', encoding='utf-8') as f:
            f.write('\\documentclass[border=1.5pt]{standalone}\n\\usepackage{tikz}\n\\input{style.tex}\n'
                    '\\begin{document}\n$\\displaystyle ' + body + '$\n\\end{document}\n')
    names = [os.path.basename(f)[:-4] for f in sorted(glob.glob(os.path.join(SRC, 'fig_*.tex')))] + list(EQ)
    for n in names:
        r = subprocess.run([tectonic, '-X', 'compile', n + '.tex'], cwd=BUILD, capture_output=True, text=True)
        if r.returncode:
            raise SystemExit(f'{n}.tex failed:\n' + '\n'.join(l for l in (r.stdout + r.stderr).splitlines() if 'error' in l))
        shutil.copy(os.path.join(BUILD, n + '.pdf'), os.path.join(FIGS, n + '.pdf'))
        subprocess.run(['pdftocairo', '-png', '-r', str(PNG_DPI), '-singlefile',
                        os.path.join(BUILD, n + '.pdf'), os.path.join(FIGS, n)], check=True)
    print(f'tex: {len(names)} figures and equations -> figures/ (PDF + {PNG_DPI} dpi PNG)')


# ---------------------------------------------------------------- PDF -> theme-aware inline SVG
def token(val):
    key = tuple(round(float(x) * 255 / 100) for x in RGB.fullmatch(val.strip()).groups())
    if key not in TOKENS:
        raise SystemExit(f'unmapped colour {key} ({val}): add it to style.tex / TOKENS')
    return TOKENS[key]


def fix_tag(m):
    tag = m.group(0)
    classes = []
    for attr, pre in (('fill', 'f'), ('stroke', 's')):
        am = re.search(r'\s' + attr + r'="([^"]+)"', tag)
        if am and am.group(1).startswith('rgb('):
            classes.append(f'{pre}-{token(am.group(1))}')
            tag = tag.replace(am.group(0), '', 1)
    if classes:
        cm = re.search(r'\sclass="([^"]*)"', tag)
        if cm:
            tag = tag.replace(cm.group(0), f' class="{cm.group(1)} {" ".join(classes)}"', 1)
        else:
            tag = re.sub(r'^<(\w+)', lambda t: f'<{t.group(1)} class="{" ".join(classes)}"', tag, count=1)
    return tag


def inline_svg(name, k, cls, label):
    svg = os.path.join(BUILD, name + '.svg')
    subprocess.run(['pdftocairo', '-svg', os.path.join(BUILD, name + '.pdf'), svg], check=True)
    s = open(svg, encoding='utf-8').read()
    s = re.sub(r'<\?xml[^>]*>\s*', '', s)
    p = name.replace('_', '')   # ids are per drawing; prefix them so 15 inline SVGs cannot collide
    s = re.sub(r'\bid="([^"]+)"', lambda m: f'id="{p}-{m.group(1)}"', s)
    s = re.sub(r'xlink:href="#([^"]+)"', lambda m: f'href="#{p}-{m.group(1)}"', s)
    s = re.sub(r'url\(#([^)]+)\)', lambda m: f'url(#{p}-{m.group(1)})', s)
    s = re.sub(r'<[a-zA-Z][^>]*>', fix_tag, s)
    head = re.match(r'<svg[^>]*>', s).group(0)
    w = float(re.search(r'width="([\d.]+)pt"', head).group(1))
    vb = re.search(r'viewBox="([^"]+)"', head).group(1)
    return s.replace(head, f'<svg xmlns="http://www.w3.org/2000/svg" class="{cls}" viewBox="{vb}" '
                           f'style="width:{w * k:.0f}px" role="img" aria-label="{label}">', 1).strip()


WRAP = ('<!doctype html>\n<html lang="en"{theme}>\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        '<style>body{{margin:0}}</style>\n</head>\n<body>\n{body}\n</body>\n</html>\n')


def build_html():
    t = open(os.path.join(SRC, 'template.html'), encoding='utf-8').read()
    for m in re.finditer(r'\{\{(fig|eq):(\w+)\|([^}]*)\}\}', t):
        kind, name, label = m.groups()
        t = t.replace(m.group(0), inline_svg(name, K_FIG if kind == 'fig' else K_EQ, 'fig' if kind == 'fig' else 'eqn', label), 1)
    assert '{{' not in t, re.findall(r'\{\{[^}]*\}\}', t)[:3]
    open(FRAGMENT, 'w', encoding='utf-8', newline='\n').write(t)
    open(HTML, 'w', encoding='utf-8', newline='\n').write(WRAP.format(theme='', body=t))
    print(f'html: {os.path.relpath(HTML, REPO)} ({os.path.getsize(HTML) / 1e6:.2f} MB); fragment {os.path.relpath(FRAGMENT, REPO)}')
    return t


def build_pdf(fragment):
    browser = tool('msedge', 'BROWSER_EXE', [r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
                                             r'C:\Program Files\Microsoft\Edge\Application\msedge.exe',
                                             r'C:\Program Files\Google\Chrome\Application\chrome.exe'])
    page = os.path.join(BUILD, 'print.html')   # light theme, fonts one level up
    open(page, 'w', encoding='utf-8', newline='\n').write(
        WRAP.format(theme=' data-theme="light"', body=fragment.replace('url(fonts/', 'url(../fonts/')))
    if os.path.exists(PDF):
        os.remove(PDF)
    subprocess.run([browser, '--headless=new', '--disable-gpu', '--no-pdf-header-footer', '--virtual-time-budget=8000',
                    f'--print-to-pdf={PDF}', 'file:///' + page.replace('\\', '/')], capture_output=True, timeout=180)
    if not os.path.exists(PDF):
        raise SystemExit('the browser wrote no PDF')
    print(f'pdf: {os.path.relpath(PDF, REPO)} ({os.path.getsize(PDF) / 1e6:.2f} MB)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', action='store_true', help='regenerate src/chartdata/ from artifacts/lb_final/ first')
    ap.add_argument('--skip-tex', action='store_true', help='reuse the PDFs already in build/')
    ap.add_argument('--no-pdf', action='store_true', help='skip the page PDF')
    a = ap.parse_args()
    if a.data:
        make_data()
    if not a.skip_tex:
        compile_tex()
    fragment = build_html()
    if not a.no_pdf:
        build_pdf(fragment)


if __name__ == '__main__':
    main()
