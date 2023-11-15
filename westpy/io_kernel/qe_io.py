import numpy as np
import json
import os

from .base_io import base_io

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
        for idx, fname in enumerate(fnames):
            with open(os.path.join(folder, fname), 'rb') as f:
                header = np.fromfile(f, dtype='int32', count=32)
                pdepg = np.fromfile(f, dtype=np.complex128, count = header[2])
                self.pdepg[idx] = pdepg

        # read pdepeig
        # Load the output data
        with open(os.path.join(folder, 'wstat.json')) as json_file :
            data = json.load(json_file)

        # Extract converged PDEP eigenvalues
        self.pdepeig = np.array(data['exec']['davitr'][-1]['ev'],dtype='f8') # -1 identifies the last iteration