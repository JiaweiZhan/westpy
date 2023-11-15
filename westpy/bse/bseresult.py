from typing import List
import json
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg.lapack as la
from westpy.units import eV
from multiprocessing import Pool

def _calc_chi_single_freq(freq: float,
                          broaden: float,
                          n_ipol,
                          nspin,
                          n_total,
                          b,
                          r,
                          zeta,
                          norm):
    degspin = 2.0 / nspin
    omeg_c = freq + broaden * 1j

    chi = np.zeros((n_ipol, n_ipol), dtype=np.complex128)

    for ip2 in range(n_ipol):
        a = np.full(n_total, omeg_c, dtype=np.complex128)
        b_l = b[ip2, :]
        c = b_l
        r_l = r

        b1, a1, c1, r1, ierr = la.zgtsv(b_l, a, c, r_l)
        assert ierr == 0

        for ip in range(n_ipol):
            chi[ip2, ip] = np.dot(zeta[ip2, ip, :], r1)
            chi[ip2, ip] *= -2.0 * degspin * norm[ip2]

    return chi

class BSEResult(object):
    def __init__(self, filename: str):
        """Parses Wbse Lanczos results.

        :param filename: Wbse output file (JSON)
        :type filename: string

        :Example:

        >>> from westpy.bse import *
        >>> wbse = BSEResult("wbse.json")
        """

        self.filename = filename

        with open(filename, "r") as f:
            res = json.load(f)

        self.nspin = res["system"]["electron"]["nspin"]
        which = res["input"]["wbse_control"]["wbse_calculation"]
        assert which in ["L", "l"]
        self.n_lanczos = res["input"]["wbse_control"]["n_lanczos"]
        pol = res["input"]["wbse_control"]["wbse_ipol"]
        if pol in ["XYZ", "xyz"]:
            self.n_ipol = 3
            self.pols = ["XX", "YY", "ZZ"]
            self.can_do = ["XX", "XY", "XZ", "YX", "YY", "YZ", "ZX", "ZY", "ZZ", "XYZ"]
        elif pol in ["XX", "xx"]:
            self.n_ipol = 1
            self.pols = ["XX"]
            self.can_do = ["XX"]
        elif pol in ["YY", "yy"]:
            self.n_ipol = 1
            self.pols = ["YY"]
            self.can_do = ["YY"]
        elif pol in ["ZZ", "zz"]:
            self.n_ipol = 1
            self.pols = ["ZZ"]
            self.can_do = ["ZZ"]

    def plotSpectrum(
        self,
        ipol: str = None,
        ispin: int = 1,
        energyRange: List[float] = [0.0, 10.0, 0.01],
        sigma: float = 0.1,
        n_extra: int = 0,
        fname: str = None,
    ):
        """Parses and plots Wbse Lanczos results.

        :param ipol: which component to compute ("XX", "XY", "XZ", "YX", "YY", "YZ", "ZX", "ZY", "ZZ", or "XYZ")
        :type ipol: string
        :param ispin: Spin channel to consider
        :type ispin: int
        :param energyRange: energy range = min, max, step (eV)
        :type energyRange: 3-dim float
        :param sigma: Broadening width (eV)
        :type sigma: float
        :param n_extra: Number of extrapolation steps
        :type n_extra: int
        :param fname: Output file name
        :type fname: string

        :Example:

        >>> from westpy.bse import *
        >>> wbse = BSEResult("wbse.json")
        >>> wbse.plotSpectrum(ipol="XYZ",energyRange=[0.0,10.0,0.01],sigma=0.1,n_extra=100000)
        """

        assert ipol in self.can_do
        assert ispin >= 1 and ispin <= self.nspin
        xmin, xmax, dx = energyRange
        assert xmax > xmin
        assert dx > 0.0
        assert sigma > 0.0
        assert n_extra >= 0

        if self.n_lanczos < 151 and n_extra > 0:
            n_extra = 0
        self.n_total = self.n_lanczos + n_extra

        self.__read_beta_zeta(ispin)
        self.__extrapolate(n_extra)

        self.r = np.zeros(self.n_total, dtype=np.complex128)
        self.r[0] = 1.0

        self.b = np.zeros((self.n_ipol, self.n_total - 1), dtype=np.complex128)
        for ip in range(self.n_ipol):
            for i in range(self.n_total - 1):
                self.b[ip, i] = -self.beta[ip, i]

        sigma_ev = sigma * eV
        n_step = int((xmax - xmin) / dx) + 1
        energyAxis = np.linspace(xmin, xmax, n_step, endpoint=True)
        chiAxis = np.zeros(n_step, dtype=np.complex128)

        freq_evs = [energy * eV for energy in energyAxis]
        chis = self.__calc_chi(freq_evs, sigma_ev)
        for ie, energy in enumerate(energyAxis):
            # calculate susceptibility for given frequency
            chi = chis[ie]

            # 1/Ry to 1/eV
            chi = chi * eV

            if self.n_ipol == 1:
                chiAxis[ie] = chi[0, 0]
            elif self.n_ipol == 3:
                if ipol == "XX":
                    chiAxis[ie] = chi[0, 0]
                if ipol == "XY":
                    chiAxis[ie] = chi[1, 0]
                if ipol == "XZ":
                    chiAxis[ie] = chi[2, 0]
                if ipol == "YX":
                    chiAxis[ie] = chi[0, 1]
                if ipol == "YY":
                    chiAxis[ie] = chi[1, 1]
                if ipol == "YZ":
                    chiAxis[ie] = chi[2, 1]
                if ipol == "ZX":
                    chiAxis[ie] = chi[0, 2]
                if ipol == "ZY":
                    chiAxis[ie] = chi[1, 2]
                if ipol == "ZZ":
                    chiAxis[ie] = chi[2, 2]
                if ipol == "XYZ":
                    chiAxis[ie] = (
                        (chi[0, 0] + chi[1, 1] + chi[2, 2]) * energy / 3.0 / np.pi
                    )

        if not fname:
            fname = f"chi_{ipol}.png"

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        dosPlot = ax.plot(energyAxis, chiAxis.imag, label=f"chi_{ipol}")

        plt.xlim([xmin, xmax])
        plt.xlabel("$\omega$ (eV)")
        if ipol == "XYZ":
            plt.ylabel("abs. coeff. (a.u.)")
        else:
            plt.ylabel("Im[$\chi$] (a.u.)")
            plt.legend()
        plt.savefig(fname, dpi=300)
        print("output written in : ", fname)
        print("waiting for user to close image preview...")
        plt.show()
        fig.clear()

    def __read_beta_zeta(self, ispin: int):
        self.norm = np.zeros(self.n_ipol, dtype=np.float64)
        self.beta = np.zeros((self.n_ipol, self.n_total), dtype=np.float64)
        self.zeta = np.zeros((self.n_ipol, 3, self.n_total), dtype=np.complex128)

        with open(self.filename, "r") as f:
            res = json.load(f)

        for ip, lp in enumerate(self.pols):
            tmp = res["output"]["lanczos"][f"K{ispin:0>6}"][lp]["beta"]
            beta_read = np.array(tmp)

            tmp = res["output"]["lanczos"][f"K{ispin:0>6}"][lp]["zeta"]
            zeta_read = np.array(tmp).reshape((3, self.n_lanczos))

            self.norm[ip] = beta_read[0]
            self.beta[ip, 0 : self.n_lanczos - 1] = beta_read[1 : self.n_lanczos]
            self.beta[ip, self.n_lanczos - 1] = beta_read[self.n_lanczos - 1]
            self.zeta[ip, :, 0 : self.n_lanczos] = zeta_read[:, 0 : self.n_lanczos]

    def __extrapolate(self, n_extra: int):
        if n_extra > 0:
            average = np.zeros(self.n_ipol, dtype=np.float64)
            amplitude = np.zeros(self.n_ipol, dtype=np.float64)

            for ip in range(self.n_ipol):
                skip = False
                counter = 0

                for i in range(150, self.n_lanczos):
                    if skip:
                        skip = False
                        continue

                    if i % 2 == 0:
                        if (
                            i != 150
                            and abs(self.beta[ip, i] - average[ip] / counter) > 2.0
                        ):
                            skip = True
                        else:
                            average[ip] = average[ip] + self.beta[ip, i]
                            amplitude[ip] = amplitude[ip] + self.beta[ip, i]
                            counter = counter + 1
                    else:
                        if (
                            i != 150
                            and abs(self.beta[ip, i] - average[ip] / counter) > 2.0
                        ):
                            skip = True
                        else:
                            average[ip] = average[ip] + self.beta[ip, i]
                            amplitude[ip] = amplitude[ip] - self.beta[ip, i]
                            counter = counter + 1

                average[ip] = average[ip] / counter
                amplitude[ip] = amplitude[ip] / counter

            for ip in range(self.n_ipol):
                for i in range(self.n_lanczos - 1, self.n_total):
                    if i % 2 == 0:
                        self.beta[ip, i] = average[ip] + amplitude[ip]
                    else:
                        self.beta[ip, i] = average[ip] - amplitude[ip]

    def __calc_chi(self, freqs,
                   broaden: float):
        with Pool() as pool:
            args = [(freq, broaden, self.n_ipol, self.nspin, self.n_total, self.b, self.r, self.zeta, self.norm) for freq in freqs]

            results = pool.starmap(_calc_chi_single_freq, args)
        return results

