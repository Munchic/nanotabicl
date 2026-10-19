"""Paper schematic of the replicate-level cell encoder (obs_model.py, default config).
Writes figures/replicate_encoder.{png,pdf}: 7 in wide (double column), 300 dpi."""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, FancyArrowPatch

ROOT = os.path.dirname(os.path.abspath(__file__))
REP, STOCK, NULL, INK, MUTED = '#0b7a70', '#7a6a55', '#7b4bb3', '#1a1a1a', '#6b6b6b'
plt.rcParams.update({'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'font.size': 7.5})

fig = plt.figure(figsize=(7, 2.3))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 140); ax.set_ylim(0, 46); ax.set_aspect('equal'); ax.axis('off')


def box(x0, x1, y0, y1, color, title=None, body=None, shape=None, fill='white', body_size=7.5):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle='round,pad=0,rounding_size=1.2',
                                fc=fill, ec=color, lw=0.9))
    xc = (x0 + x1) / 2
    if title:
        ax.text(xc, y1 + 1.0, title, ha='center', va='bottom', fontsize=7, color=INK, fontweight='bold')
    if body:
        ax.text(xc, (y0 + y1) / 2, body, ha='center', va='center', fontsize=body_size, color=INK, linespacing=1.5)
    if shape:
        ax.text(xc, y0 - 1.0, shape, ha='center', va='top', fontsize=6.3, color=MUTED)


def arrow(p, q, color=INK, style='-|>', ls='-', rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=6, lw=0.8, color=color, ls=ls,
                                 connectionstyle=f'arc3,rad={rad}', shrinkA=0, shrinkB=0))


# ----- input cell: K replicate slots, n valid
for k in range(6):
    on = k < 4
    ax.add_patch(Rectangle((2.5, 13.5 + 2.6 * k), 2.2, 2.2, fc=REP if on else 'white', ec=REP if on else MUTED,
                           lw=0.6, ls='-' if on else (0, (2, 1.5))))
ax.text(3.6, 30.6, 'cell', ha='center', va='bottom', fontsize=6.5, color=MUTED)
ax.text(3.6, 12.3, '$r_k,\\,m_k$', ha='center', va='top', fontsize=7)

# ----- replicate branch (top row)
Y0, Y1 = 25, 36
ax.add_patch(FancyBboxPatch((11.5, 21.0), 67.5, 20.2, boxstyle='round,pad=0,rounding_size=1.5',
                            fc='none', ec=REP, lw=0.7, ls=(0, (3, 2))))
ax.text(12.5, 42.2, 'ReplicateEncoder  (cells with $n \\geq 2$; others give $e = 0$)', ha='left', va='bottom',
        fontsize=6.8, color=REP)
box(14, 30, Y0, Y1, REP, 'per-replicate', '$\\phi([z_k,\\, z_k - \\bar z])$\n$z = (r - \\mu_j)/\\sigma_j$',
    '$(M, K, 32)$', body_size=6.8)
box(34, 50, Y0, Y1, REP, 'set self-attn', '$\\mathrm{MHA}_{4\\times 8}$\nmask $m$', '$(M, K, 32)$')
box(54, 76, Y0, Y1, REP, 'pool + null slot', None, '$(M, 32)$')
# pooling sketch: seed query s attends to null slot + valid replicates (widths ~ illustrative weights)
sx, sy, ty = 65, 33.6, 27.6
toks = [(57.5, NULL, 0.10, '$\\varnothing$'), (61.2, REP, 0.28, ''), (64.6, REP, 0.31, ''),
        (68.0, REP, 0.27, ''), (71.4, REP, 0.04, '')]
for x, c, a, _ in toks:
    ax.plot([sx, x], [sy - 1.1, ty + 1.1], color=c, lw=0.4 + 6 * a, solid_capstyle='round', alpha=0.85, zorder=2)
for x, c, _, lab in toks:
    ax.add_patch(Circle((x, ty), 1.1, fc=c, ec='none', zorder=3))
    if lab:
        ax.text(x, ty - 0.05, lab, ha='center', va='center', fontsize=6, color='white', zorder=4)
ax.add_patch(Circle((74.4, ty), 1.1, fc='white', ec=MUTED, lw=0.5, ls=(0, (1.5, 1.2)), zorder=3))
ax.add_patch(Circle((sx, sy), 1.2, fc='white', ec=REP, lw=0.9, zorder=3))
ax.text(sx, sy - 0.1, '$s$', ha='center', va='center', fontsize=7, zorder=4)

box(81, 97, Y0, Y1, REP, 'append count', '$e = [\\mathrm{LN}(p);\\, \\log n]$', '$(B, R, C, 33)$', body_size=6.8)
box(101, 115, Y0, Y1, REP, 'group, project', '$U[e_j; e_{j+1}; e_{j+3}]$\n$U_0 = 0$', '$(B, R, C, 32)$',
    body_size=6.3)
for a_, b_ in [(30, 34), (50, 54), (76, 81), (97, 101)]:
    arrow((a_, (Y0 + Y1) / 2), (b_, (Y0 + Y1) / 2), REP)
arrow((5.6, 22.0), (14, (Y0 + Y1) / 2), REP, rad=-0.2)

# ----- stock path (bottom row)
S0, S1 = 4, 13
box(14, 30, S0, S1, STOCK, None, '$\\bar x = \\frac{1}{n}\\,\\Sigma_k\\, m_k r_k$', '$(B, R, C)$', body_size=6.8)
box(81, 97, S0, S1, STOCK, None, '$W_x[\\tilde x_j; \\tilde x_{j+1}; \\tilde x_{j+3}]$', '$(B, R, C, 32)$',
    body_size=6.3)
ax.text(22, S1 + 1.0, 'cell mean', ha='center', va='bottom', fontsize=7, color=INK, fontweight='bold')
ax.text(89, S1 + 1.0, 'stock embedding', ha='center', va='bottom', fontsize=7, color=INK, fontweight='bold')
arrow((5.6, 15.0), (14, (S0 + S1) / 2), STOCK, rad=0.2)
arrow((30, (S0 + S1) / 2), (81, (S0 + S1) / 2), STOCK)
ax.text(55.5, (S0 + S1) / 2 + 0.9, 'standardize ($\\mu_j, \\sigma_j$ from context rows)', ha='center', va='bottom',
        fontsize=6.3, color=MUTED)

# ----- merge + backbone
cx, cy = 119.5, 19.5
ax.add_patch(Circle((cx, cy), 2.0, fc='white', ec=INK, lw=0.8))
ax.text(cx, cy - 0.15, '+', ha='center', va='center', fontsize=9)
arrow((115, (Y0 + Y1) / 2), (cx, cy + 2.0), REP, rad=-0.35)
arrow((97, (S0 + S1) / 2), (cx - 1.0, cy - 1.75), STOCK, rad=-0.25)
box(124.5, 138.5, 14, 25, INK, 'backbone', 'TF$_\\mathrm{col}$, TF$_\\mathrm{row}$,\nTF$_\\mathrm{icl}$', 'unchanged',
    fill='#f2f2f2', body_size=7)
arrow((cx + 2.0, cy), (124.5, cy), INK)

os.makedirs(os.path.join(ROOT, 'figures'), exist_ok=True)
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(ROOT, 'figures', f'replicate_encoder.{ext}'), dpi=300)
print('wrote figures/replicate_encoder.{png,pdf}')
