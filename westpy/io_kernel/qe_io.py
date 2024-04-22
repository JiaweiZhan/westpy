import numpy as np
import json
import os, sys
from typing import List
from concurrent.futures import ThreadPoolExecutor

from .base_io import base_io

HD_LENGTH = 32
HD_VERSION = 210405
HD_ID_VERSION = 0
HD_ID_LITTLE_ENDIAN = 1
HD_ID_DIMENSION = 2

class qe_io(base_io):
    def __init__(self,
                 wfc_fname=None,
                 wstat_folder=None):
        super().__init__()
        self.wfc_name = wfc_fname
        self.wstat_folder = wstat_folder

        if self.wfc_name:
            if isinstance(self.wfc_name, str):
                self.read_wfc(str(self.wfc_name))
            else:
                raise TypeError

        if self.wstat_folder:
            if isinstance(self.wstat_folder, str):
                self.read_wstat(str(self.wstat_folder))
            else:
                raise TypeError

    def read_wfc(self,
                 fname: str,
                 read_evc: bool=True):
        self.fname = fname
        with open(self.fname, 'rb') as f:
            # Moves the cursor 4 bytes to the right
            f.seek(4)

            self.ik = np.fromfile(f, dtype='int32', count=1)[0]
            self.xk = np.fromfile(f, dtype='float64', count=3)
            self.ispin = np.fromfile(f, dtype='int32', count=1)[0]
            self.gamma_only = bool(np.fromfile(f, dtype='int32', count=1)[0])
            self.scalef = np.fromfile(f, dtype='float64', count=1)[0]

            # Move the cursor 8 byte to the right
            f.seek(8, 1)

            self.ngw = np.fromfile(f, dtype='int32', count=1)[0]
            self.igwx = np.fromfile(f, dtype='int32', count=1)[0]
            self.npol = np.fromfile(f, dtype='int32', count=1)[0]
            self.nbnd = np.fromfile(f, dtype='int32', count=1)[0]

            # Move the cursor 8 byte to the right
            f.seek(8, 1)

            self.b1 = np.fromfile(f, dtype='float64', count=3)
            self.b2 = np.fromfile(f, dtype='float64', count=3)
            self.b3 = np.fromfile(f, dtype='float64', count=3)

            f.seek(8,1)
            
            self.mill = np.fromfile(f, dtype='int32', count=3*self.igwx)
            self.mill = self.mill.reshape( (self.igwx, 3) ) 

            if read_evc:
                self.evc = np.zeros( (self.nbnd, self.npol * self.igwx), dtype="complex128")

                f.seek(8,1)
                for i in range(self.nbnd):
                    self.evc[i,:] = np.fromfile(f, dtype='complex128', count=self.npol * self.igwx)
                    f.seek(8, 1)
        volume_resiprocal = np.abs(np.dot(np.cross(self.b1, self.b2), self.b3))
        self.omega = np.power(2 * np.pi, 3) / (volume_resiprocal)
        self.a1 = (2 * np.pi * (np.cross(self.b2, self.b3) / volume_resiprocal)).tolist()
        self.a2 = (2 * np.pi * (np.cross(self.b3, self.b1) / volume_resiprocal)).tolist()
        self.a3 = (2 * np.pi * (np.cross(self.b1, self.b2) / volume_resiprocal)).tolist()

    @staticmethod
    def read_pdep(fname):
        with open(fname, 'rb') as f:
            header = np.fromfile(f, dtype='int32', count=32)
            pdepg = np.fromfile(f, dtype=np.complex128, count=header[2])
        return pdepg

    @staticmethod
    def pdep_index(fname):
        base_name = os.path.basename(fname)
        file_name, _ = os.path.splitext(base_name)
        return int(file_name.split("E")[-1])

    def read_wstat(self,
                   folder: str,
                  ):
        fnames_all = os.listdir(folder)
        fnames = []
        for fname in fnames_all:
            if '.dat' in fname:
                fnames.append(fname)

        # read pdepg
        pdep_index = [self.pdep_index(fname) for fname in fnames]
        fnames_cp = [""] * len(fnames)
        for idx, fname in zip(pdep_index, fnames):
            fnames_cp[idx - 1] = fname
        fnames = fnames_cp
        self.pdepg = np.zeros( (len(fnames), len(self.mill)), dtype="complex128")
        fnames = [os.path.join(folder, fname) for fname in fnames]
        with ThreadPoolExecutor() as executor:
            results = executor.map(lambda fname: self.read_pdep(fname), fnames)
        for idx, pdepg in enumerate(results):
            self.pdepg[idx] = pdepg

        # read pdepeig
        # Load the output data
        with open(os.path.join(folder, 'wstat.json')) as json_file :
            data = json.load(json_file)

        # Extract converged PDEP eigenvalues
        self.pdepeig = np.array(data['exec']['davitr'][-1]['ev'],dtype='f8') # -1 identifies the last iteration


    @staticmethod
    def write_pdep(pdep_val: np.ndarray,
                   fname: str):
        header = np.zeros(HD_LENGTH, dtype='int32')
        header[HD_ID_VERSION] = HD_VERSION
        header[HD_ID_DIMENSION] = pdep_val.size
        header[HD_ID_LITTLE_ENDIAN] = 1 if sys.byteorder == 'little' else 0
        with open(fname, 'wb') as f:
            f.write(header.tobytes())
            f.write(pdep_val.tobytes())

    def write_summary(self,
                      pdepeig: np.ndarray,
                      pdep_fnames: List[str],
                      fname: str='summary_westpy.json',
                      ):
        data = {}
        data["dielectric_matrix"] = {}
        data['dielectric_matrix']['domain'] = {}
        data['dielectric_matrix']['domain']['a1'] = self.a1
        data['dielectric_matrix']['domain']['a2'] = self.a2
        data['dielectric_matrix']['domain']['a3'] = self.a3
        data['dielectric_matrix']['domain']['b1'] = self.b1.tolist()
        data['dielectric_matrix']['domain']['b2'] = self.b2.tolist()
        data['dielectric_matrix']['domain']['b3'] = self.b3.tolist()
        data['dielectric_matrix']['pdep'] = [{}]
        data['dielectric_matrix']['pdep'][0]['iq'] = 1
        data['dielectric_matrix']['pdep'][0]['q'] = [0.0, 0.0, 0.0]
        data['dielectric_matrix']['pdep'][0]['eigenval'] = pdepeig.tolist()
        data['dielectric_matrix']['pdep'][0]['eigenvec'] = pdep_fnames
        with open(fname, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def write_wstat(self,
                    pdepeig: np.ndarray,
                    pdepvec: np.ndarray,
                    prefix: str = "westpy",
                    eig_mat : str = 'chi',
                    ):
        assert eig_mat in ['chi', 'chi_0'], "eig_mat must be either 'chi' or 'chi_0'"
        if eig_mat == 'chi':
            pdepeig = pdepeig / (pdepeig + 1.0)
        outFolder = prefix + ".wstat.save"
        if not os.path.exists(outFolder):
            os.makedirs(outFolder)
        npdep = pdepvec.shape[0]
        fnames = []
        for i in range(npdep):
            fname = "Q" + str(1).zfill(9) + "E" + str(i + 1).zfill(9) + ".dat"
            fnames.append(fname)

        def write_wstat_helper(vec, fname, outFolder):
            self.write_pdep(vec, os.path.join(outFolder, fname))

        with ThreadPoolExecutor() as executor:
            executor.map(lambda vecfname: write_wstat_helper(*vecfname, outFolder=outFolder),
                         zip(pdepvec, fnames))
        self.write_summary(pdepeig, fnames, os.path.join(outFolder, 'summary.json'))
