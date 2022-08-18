from typing import Dict, List, Tuple, Optional, Union
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import copy
from lxml import etree
from westpy.qdet.misc import find_indices, parse_one_value, find_index
from westpy.qdet.misc import VData, rydberg_to_hartree, ev_to_hartree, hartree_to_ev
from westpy.qdet.heff import Heff
from westpy.qdet.symm import PointGroup, PointGroupRep
from westpy.qdet.west_output import WstatOutput


class QDETResult(object):

    ev_thr = 0.001 * ev_to_hartree  # threshold for determining degenerate states
    occ_thr = 0.001  # threshold for determining equal occupations

    def __init__(
        self,
        filename: str,
        point_group: Optional[PointGroup] = None,
        symmetrize: Dict[str, bool] = {},
    ):
        """Parser for Quantum Defect Embedding Theory (QDET) calculations.

        Args:
            filename: name of JSON file that contains the output of WEST
            calculation.
            point_group: point group of the system.
            symmetrize: arguments for symmetrization function of Heff.
        """
        self.filename = filename

        # read basic parameters from JSON file
        self.nspin, self.npair, self.basis = self.__read_parameters_from_JSON(filename)

        # read occupation from file
        self.occupation = self.__read_occupation_from_JSON(filename)

        # read one- and two-body terms from JSON file
        self.h1e, self.eri = self.__read_matrix_elements_from_JSON(filename)

        # determine point-group representation
        self.point_group = point_group
        if self.point_group is None:
            point_group_rep = None
        else:
            orbitals = [
                VData(
                    f"{self.path}/west.westpp.save/wfcK000001B{i:06d}.cube",
                    normalize="sqrt",
                )
                for i in self.ks_projectors
            ]
            point_group_rep, orbital_symms = self.point_group.compute_rep_on_orbitals(
                orbitals, orthogonalize=True
            )
        self.h1e = self.h1e / hartree_to_ev
        self.eri = self.eri / hartree_to_ev

        # generate effective Hamiltonian
        self.heff = Heff(self.h1e, self.eri, point_group_rep=point_group_rep)

        self.heff.symmetrize(**symmetrize)

    def __str__(self):
        """Print a summary of QDET calculation."""
        string = "---------------------------------------------------------------\n"
        string += "CGW Results General Info\n"
        string += f"path: {self.path}\n"
        string += f"nspin = {self.nspin}, nel = {self.nel}, nproj = {len(self.bases)}\n"
        if self.point_group is not None:
            string += f"point group: {self.point_group.name}\n"
        string += f"ks_projectors: {self.basis}\n"

        string += "occupations:" + str(self.occupation) + "\n"
        string += "---------------------------------------------------------------\n"

        return string

    def __read_parameters_from_JSON(self, filename):
        """
        Read basic calculation parameters from JSON file.
        """

        with open(filename, "r") as f:
            raw_ = json.load(f)

        indexmap = np.array(raw_["output"]["Q"]["indexmap"], dtype=int)

        npair = len(indexmap)
        nspin = int(raw_["system"]["electron"]["nspin"])
        bands = np.array(raw_["input"]["wfreq_control"]["qp_bands"], dtype=int)

        return nspin, npair, bands

    def __read_occupation_from_JSON(self, filename):
        """
        Read DFT occupation from JSON file.
        """
        with open(filename, "r") as f:
            raw_ = json.load(f)

        occ_ = np.zeros((self.nspin, len(self.basis)))

        for ispin in range(self.nspin):
            string1 = "K" + format(ispin + 1, "06d")
            occ_[ispin, :] = np.array(
                raw_["output"]["Q"][string1]["occupation"], dtype=float
            )

        return occ_

    def __read_matrix_elements_from_JSON(self, filename):
        """
        Read one-body and two-body terms from JSON file.
        """

        with open(filename, "r") as f:
            raw_ = json.load(f)

        indexmap = np.array(raw_["output"]["Q"]["indexmap"], dtype=int)

        # allocate one- and two-body terms in basis of pairs of KS states
        eri_pair = np.zeros((self.nspin, self.nspin, self.npair, self.npair))
        h1e_pair = np.zeros((self.nspin, self.npair))

        # read one-body terms from file
        for ispin in range(self.nspin):
            string1 = "K" + format(ispin + 1, "06d")
            h1e_pair[ispin, :] = np.array(raw_["qdet"]["h1e"][string1], dtype=float)

        # read two-body terms from file
        for ispin1 in range(self.nspin):
            string1 = "K" + format(ispin1 + 1, "06d")
            for ispin2 in range(self.nspin):
                string2 = "K" + format(ispin2 + 1, "06d")

                for ipair in range(self.npair):
                    string3 = "pair" + format(ipair + 1, "06d")
                    eri_pair[ispin1, ispin2, ipair, :] = np.array(
                        raw_["qdet"]["eri"][string1][string2][string3], dtype=float
                    )

        # unfold one-body terms from pair basis to Kohn-Sham basis
        h1e = np.zeros((self.nspin, len(self.basis), len(self.basis)))
        for ispin in range(self.nspin):
            for ipair in range(len(indexmap)):
                i, j = indexmap[ipair]
                h1e[ispin, i - 1, j - 1] = h1e_pair[ispin, ipair]
                h1e[ispin, j - 1, i - 1] = h1e_pair[ispin, ipair]

        # unfold two-body terms from pair to Kohn-Sham basis
        eri = np.zeros(
            (
                self.nspin,
                self.nspin,
                len(self.basis),
                len(self.basis),
                len(self.basis),
                len(self.basis),
            )
        )
        for ispin in range(self.nspin):
            for jspin in range(self.nspin):
                for ipair in range(len(indexmap)):
                    i, j = indexmap[ipair]
                    for jpair in range(len(indexmap)):
                        k, l = indexmap[jpair]
                        eri[ispin, jspin, i - 1, j - 1, k - 1, l - 1] = eri_pair[
                            ispin, jspin, ipair, jpair
                        ]
                        eri[ispin, jspin, i - 1, j - 1, l - 1, k - 1] = eri_pair[
                            ispin, jspin, ipair, jpair
                        ]
                        eri[ispin, jspin, j - 1, i - 1, k - 1, l - 1] = eri_pair[
                            ispin, jspin, ipair, jpair
                        ]
                        eri[ispin, jspin, j - 1, i - 1, l - 1, k - 1] = eri_pair[
                            ispin, jspin, ipair, jpair
                        ]

                        eri[ispin, jspin, k - 1, l - 1, i - 1, j - 1] = eri_pair[
                            ispin, jspin, ipair, jpair
                        ]
                        eri[ispin, jspin, k - 1, l - 1, j - 1, i - 1] = eri_pair[
                            ispin, jspin, ipair, jpair
                        ]
                        eri[ispin, jspin, l - 1, k - 1, i - 1, j - 1] = eri_pair[
                            ispin, jspin, ipair, jpair
                        ]
                        eri[ispin, jspin, l - 1, k - 1, j - 1, i - 1] = eri_pair[
                            ispin, jspin, ipair, jpair
                        ]
        return h1e, eri

    def write(self, *args):

        data = ""
        for i in args:
            data += str(i)
            data += " "
        data = data[:-1]
        print(data)

    def solve(
        self, nelec: Tuple = None, nroots: int = 10, verbose: bool = True
    ) -> Dict:
        """Build effective Hamiltonians for given active space.

        The highest level function of CGWResults class. Call self.make_heff to build
        effective Hamiltonians for given set of W. Can run FCI calculations in place.

        Args:
            nelec: # of electrons in each spin-channel
            nroots: # of roots for FCI calculations.
            verbose: if True, write detailed info for FCI calculations.
        """
        basis_indices = self.basis
        basis_labels = [""] * len(basis_indices)

        # determine number of electrons from occupations
        if nelec == None:
            if self.nspin == 1:
                nel = np.sum(self.occupation)
                nelec = (int(round(nel)) // 2, int(round(nel)) // 2)
            else:
                nel1 = np.sum(self.occupation[0, :])
                nel2 = np.sum(self.occupation[1, :])
                nelec = (int(nel1), int(nel2))

        # diagonalize effective Hamiltonian
        fcires = self.heff.FCI(nelec=nelec, nroots=nroots)

        if verbose:

            self.write(
                "==============================================================="
            )
            self.write("Building effective Hamiltonian...")
            self.write(f"nspin: {self.nspin}")
            self.write(f"occupations: {self.occupation[:]}")
            self.write(
                "==============================================================="
            )

            self.write("-----------------------------------------------------")

            self.write(f"{'#':>2}  {'ev':>5} {'term':>4} diag[1RDM - 1RDM(GS)]")
            self.write(f"{'':>15}" + " ".join(f"{b:>4}" for b in self.basis))
            ispin = 0
            if self.point_group is not None:
                self.write(
                    f"{'':>15}"
                    + " ".join(f"{s.partition('(')[0]:>4}" for s in orbital_symms)
                )
            for i, (ev, mult, symm, ex) in enumerate(
                zip(
                    fcires["evs"],
                    fcires["mults"],
                    fcires["symms_maxproj"],
                    fcires["excitations"],
                )
            ):
                symbol = f"{int(round(mult))}{symm.partition('(')[0]}"
                exstring = " ".join(f"{e:>4.1f}" for e in ex)
                self.write(f"{i:>2}  {ev:.3f} {symbol:>4} {exstring}")

            self.write("-----------------------------------------------------")

        return fcires
