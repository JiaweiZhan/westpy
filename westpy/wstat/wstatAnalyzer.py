import numpy as np
from westpy import qe_io
from westpy import wfc_g2r

class wstatAnalyzer:
    def __init__(self,
                 qe_container: qe_io,
                 fftw,
                 ):
        self.qe_container = qe_container
        self.eigvec = np.array([])
        self.fftw = list(fftw)

    def epsilon_diag(self,
                     ):
        g_norm = np.linalg.norm(self.qe_container.mill, axis=-1, keepdims=False)
        eps_diag = 1 + np.sum(((self.qe_container.pdepeig) / (1 - self.qe_container.pdepeig))[:, None] * np.abs(self.qe_container.pdepg) ** 2, axis=0)

        g_norm, eps_diag = zip(*(sorted(zip(g_norm, eps_diag), key=lambda x: x[0])))
        return g_norm, eps_diag

    def scale_pdep(self,
                   ):
        b = np.array([self.qe_container.b1, self.qe_container.b2, self.qe_container.b3])
        g_vec = self.qe_container.mill @ b      # [nmill, 3]
        g_vec_norm = np.linalg.norm(g_vec, axis=-1)
        g_vec_norm[0] = 1
        scaled_pdep_g = self.qe_container.pdepg / g_vec_norm[None, :] * np.sqrt(4 * np.pi)
        self.eigvec = scaled_pdep_g

    def wfc_g2r(self,
                ):
        eigvec_r = wfc_g2r(self.eigvec, self.fftw, self.qe_container.mill)
        return eigvec_r

    def screened_coulomb_interaction_r(self,
                                       point_1,
                                       point_2,
                                       scaled_pdep_r: np.ndarray, # [npdep, fftw[0], fftw[1], fftw[2]]
                                       eps_infty: float = 1
                                       ):
        a = np.array([self.qe_container.a1, self.qe_container.a2, self.qe_container.a3])
        point_1 = np.array(point_1)
        point_2 = np.array(point_2)
        assert np.all(point_1 >= 0)
        assert np.all(point_1 < np.array(self.fftw))
        assert np.all(point_2 >= 0)
        assert np.all(point_2 < np.array(self.fftw))
        # Prepare indices for advanced indexing
        indices_1 = (slice(None),) + tuple(point_1)
        indices_2 = (slice(None),) + tuple(point_2)

        omega = np.abs(np.dot(np.cross(a[0], a[1]), a[2]))
        bci_r = self.bared_coulomb_interaction_r(point_1, point_2)
        head = 2 / np.pi * np.power(6 * np.pi ** 2 / omega, 1 / 3) * (1 - eps_infty) / eps_infty

        return 1.0 / omega * np.sum(scaled_pdep_r[indices_1] * self.qe_container.pdepeig / (1 - self.qe_container.pdepeig) * scaled_pdep_r[indices_2]) + bci_r + head

    def bared_coulomb_interaction_r(self,
                                    point_1,
                                    point_2,
                                    ):
        a = np.array([self.qe_container.a1, self.qe_container.a2, self.qe_container.a3])
        point_1 = np.array(point_1)
        point_2 = np.array(point_2)
        assert np.all(point_1 >= 0)
        assert np.all(point_1 < np.array(self.fftw))
        assert np.all(point_2 >= 0)
        assert np.all(point_2 < np.array(self.fftw))

        if np.all(point_1 == point_2):
            return 0

        diff = np.array(point_1 - point_2, dtype=np.double)
        diff[0] = ((diff[0] + self.fftw[0] // 2) % self.fftw[0] - self.fftw[0] // 2) * 1.0 / self.fftw[0]
        diff[1] = ((diff[1] + self.fftw[1] // 2) % self.fftw[1] - self.fftw[1] // 2) * 1.0 / self.fftw[1]
        diff[2] = ((diff[2] + self.fftw[2] // 2) % self.fftw[2] - self.fftw[2] // 2) * 1.0 / self.fftw[2]
        dis = np.linalg.norm(diff[None, :] @ a)
        return 1.0 / dis
