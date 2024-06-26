import os

os.environ["OMP_NUM_THREADS"] = "1"  # For OpenMP
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # For OpenBLAS
os.environ["MKL_NUM_THREADS"] = "1"  # For MKL

from westpy.units import *
from westpy.utils import *
from westpy.atom import *
from westpy.cell import *
from westpy.vdata import *
from westpy.geometry import *
from westpy.groundState import *
from westpy.dataContainer import *
from westpy.electronicStructure import *
from westpy.session import *
from westpy.lifetime import *
from westpy.qdet import *
from westpy.bse import *
from westpy.io_kernel import *
from westpy.overlap_matrix import *
from westpy.wstat import *
from westpy.response import *

__version__ = "5.5.0.pdep"


def header():
    """Prints welcome header."""
    import datetime

    print(" ")
    print(" _    _ _____ _____ _____            ")
    print("| |  | |  ___/  ___|_   _|           ")
    print("| |  | | |__ \ `--.  | |_ __  _   _  ")
    print("| |/\| |  __| `--. \ | | '_ \| | | | ")
    print("\  /\  / |___/\__/ / | | |_) | |_| | ")
    print(" \/  \/\____/\____/  \_/ .__/ \__, | ")
    print("                       | |     __/ | ")
    print("                       |_|    |___/  ")
    print(" ")
    print("WEST version     : ", __version__)
    print("Today            : ", datetime.datetime.today())


header()
