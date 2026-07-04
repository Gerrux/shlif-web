"""Phase label conventions and design-system colours.

The pipeline segments a reflected-light image into three *reflectance* phases,
plus talc as a sub-phase of the matrix and two intergrowth *types* derived from
how sulfide and magnetite interlock.

Reflectance ordering in reflected light (bright -> dark):
    sulfide (bright cream)  >  magnetite (mid grey)  >  silicate/talc matrix (dark)
"""

from __future__ import annotations

# ---- reflectance phase labels (values in a label map) ----
MATRIX = 0     # dark silicate / talc-bearing gangue
MAGNETITE = 1  # mid-grey oxide, the "серая фаза" that replaces sulfide
SULFIDE = 2    # bright reflective ore phase

PHASE_NAMES = {
    MATRIX: "matrix",
    MAGNETITE: "magnetite",
    SULFIDE: "sulfide",
}

# ---- intergrowth types (of sulfide) ----
INTERGROWTH_NORMAL = "normal"  # обычные срастания — large clean sulfide
INTERGROWTH_FINE = "fine"      # тонкие срастания — sulfide densely laced with magnetite

# ---- ore classes (final verdict) ----
ORE_ORDINARY = "ordinary"        # рядовая руда
ORE_HARD = "hard"                # труднообогатимая руда
ORE_TALCOSE = "talcose"          # оталькованная руда
ORE_REVIEW = "review"            # на проверку (low confidence)

ORE_CLASS_RU = {
    ORE_ORDINARY: "рядовая руда",
    ORE_HARD: "труднообогатимая руда",
    ORE_TALCOSE: "оталькованная руда",
    ORE_REVIEW: "на проверку",
}

# ---- design-system "Шлиф" colours as RGB (approx. of the oklch tokens) ----
COLOR_SULFIDE = (201, 180, 95)    # brass — sulfides / brand
COLOR_MAGNETITE = (150, 160, 182)  # steel grey — magnetite
COLOR_TALC = (79, 143, 240)       # phase-talc — blue
COLOR_NORMAL = (63, 174, 107)     # phase-normal — green (обычные срастания)
COLOR_FINE = (224, 85, 78)        # phase-fine — red (тонкие срастания)
COLOR_MATRIX = None               # matrix left un-tinted
