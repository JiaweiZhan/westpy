from westpy import qe_io
import numpy as np
from typing import Tuple
from loguru import logger
from westpy import wfc_g2r

class Response:
    def __init__(self,
                 qe_container: qe_io,
                 fftw: Tuple[int]):
        self.qe_container = qe_container
        self.chi_eigval = self.qe_container.pdepeig / (1 - self.qe_container.pdepeig)
        self.chi_eigvec = self.compute_eigvec()
        self.chi_eigvec_r = None
        self.fftw = fftw

    def compute_eigvec(self,
                       ):
        b = np.array([self.qe_container.b1, self.qe_container.b2, self.qe_container.b3])
        g_vec = self.qe_container.mill @ b      # [nmill, 3]
        g_vec_norm = np.linalg.norm(g_vec, axis=-1)
        return self.qe_container.pdepg * g_vec_norm[None, :] / np.sqrt(4 * np.pi)

    def compute_response(self,
                         perturbation: np.ndarray,
                         ):
        if len(perturbation.shape) == 3:
            perturbation = perturbation[None, :, :, :]

        if not np.all(np.array(self.fftw) == perturbation.shape[1:]):
            logger.error(f"perturbation shape error: {perturbation.shape} -- {self.fftw}")
            raise RuntimeError

        # ----- real space way ----

        # if self.chi_eigvec_r is None:
        #     self.chi_eigvec_r = wfc_g2r(self.chi_eigvec, list(self.fftw), self.qe_container.mill).real
        # prefactor = np.mean(self.chi_eigvec_r * perturbation[None, :, :, :], axis=(1, 2, 3)) * self.qe_container.omega
        # return np.sum((self.chi_eigval * prefactor)[:, None, None, None] * self.chi_eigvec_r, axis=0)
        # -------------------------

        # -- reciprocal space way --
        g1, g2, g3 = self.qe_container.mill.T

        perturbation_g = np.fft.fftn(perturbation, norm='forward', axes=(-3, -2, -1))[:, g1, g2, g3]
        prefactor = self.qe_container.omega * np.sum(self.chi_eigvec.conj()[None, :, :] * perturbation_g.conj()[:, None, :], axis=-1) # [nbatch, neig]
        deltarho_g = np.sum((self.chi_eigval[None, :] * prefactor)[:, :, None] * self.chi_eigvec[None, :, :], axis=1)
        deltarho_r = wfc_g2r(deltarho_g, list(self.fftw), self.qe_container.mill).real
        return deltarho_r
        # -------------------------
