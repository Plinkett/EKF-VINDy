import matplotlib as mpl

"""
Plotting utilities directory. Tries to set LaTex at the beginning.
"""
# TODO: add exception handling for missing LaTeX installation

def safe_set_latex():
    latex_available = False
    try:
        mpl.rcParams.update({
            "text.usetex": True,                     # Use LaTeX for all text rendering
            "font.family": "serif",                  # Default LaTeX font
            "text.latex.preamble": r"\usepackage{amsmath, amssymb}"  # Optional: extra LaTeX packages
        })
        print("LaTeX is installed. Using LaTeX for rendering.")
        latex_available=True
    
    except RuntimeError:
        print("LaTeX is not installed. Falling back to default matplotlib settings.")
        mpl.rcParams.update({
            "text.usetex": False,                    # Use default matplotlib settings
            "font.family": "sans-serif"              # Default sans-serif font
        })
    return latex_available

latex_available = safe_set_latex()