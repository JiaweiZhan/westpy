import numpy as np
from typing import List
from loguru import logger

from westpy import cupy_is_available

use_gpu = False
if cupy_is_available():
    import cupy as cp
    use_gpu = True
    total_gpu_memory, free_gpu_memory = cp.cuda.Device().mem_info
    logger.info(f"GPU free memory: {free_gpu_memory / 10 ** 6:^g} MB")

class overlapMatrix:
    def __init__(self,
                 wfc: np.ndarray,   # [nwfc, ndim]
                 mills: np.ndarray, # [nwfc, 3]
                 fftw: List[int],   # [3]
                 gpu: bool = True,
                 ):
        self.use_gpu = use_gpu and gpu
        self.wfc = wfc
        self.mills = mills
        self.fftw = list(fftw)
        self.overlap_matrix = None
        self.localization = None
        self.transformer = None

    def wfc_g2r(self,
                gamma_only: bool = True,
                batch_size: int = 0):
        nwfc, _ = self.wfc.shape
        wfc_g = np.zeros([nwfc] + self.fftw, dtype=self.wfc.dtype)
        mill_x, mill_y, mill_z = self.mills.T

        wfc_indices = np.arange(nwfc)[:, None]
        wfc_g[wfc_indices, mill_x, mill_y, mill_z] = self.wfc
        if gamma_only:
            wfc_g[wfc_indices, - mill_x[1:], - mill_y[1: ], - mill_z[1: ]] = self.wfc[:, 1:].conj()

        if self.use_gpu:
            wfc_r = np.zeros([nwfc] + self.fftw, dtype=self.wfc.dtype)
            if batch_size == 0:
                batch_size = int(free_gpu_memory / np.prod(self.fftw) / 16 / 5)
                if batch_size < 1:
                    logger.error(f"batch_size smaller than one! Free memory: {free_gpu_memory / 10 ** 6:^g} MB")
                    raise RuntimeError

            for start in range(0, nwfc, batch_size):
                end = min(start + batch_size, nwfc)
                wfc_r_batch = cp.fft.ifftn(cp.asarray(wfc_g[start:end]),
                                           axes=list(range(1, len(self.fftw) + 1)),
                                           norm='ortho')

                # Store the results in the corresponding part of wfc_r_total
                wfc_r[start:end] = wfc_r_batch.get()
        else:
            wfc_r = np.fft.ifftn(wfc_g,
                                 axes=list(range(1, len(self.fftw) + 1)),
                                 norm='ortho')
        return wfc_r

    def compute_localization(self,
                             region_a: List = [0, 1],
                             region_b: List = [0, 1],
                             region_c: List = [0, 1],
                             chunk_size: int = 0,
                             ):
        nwfc, _ = self.wfc.shape
        if self.use_gpu:
            self.overlap_matrix = cp.zeros((nwfc, nwfc), dtype = self.wfc.dtype)
        else:
            self.overlap_matrix = np.zeros((nwfc, nwfc), dtype = self.wfc.dtype)
        wfc_r = self.wfc_g2r()

        # Generating normalized grids for each dimension
        if self.use_gpu:
            ix_grid, iy_grid, iz_grid = cp.meshgrid(cp.arange(self.fftw[0]) / self.fftw[0],
                                                    cp.arange(self.fftw[1]) / self.fftw[1],
                                                    cp.arange(self.fftw[2]) / self.fftw[2],
                                                    indexing='ij')
        else:
            ix_grid, iy_grid, iz_grid = np.meshgrid(np.arange(self.fftw[0]) / self.fftw[0],
                                                    np.arange(self.fftw[1]) / self.fftw[1],
                                                    np.arange(self.fftw[2]) / self.fftw[2],
                                                    indexing='ij')

        projector = ((ix_grid >= region_a[0]) & (ix_grid < region_a[1]) &
                     (iy_grid >= region_b[0]) & (iy_grid < region_b[1]) &
                     (iz_grid >= region_c[0]) & (iz_grid < region_c[1]))

        wfc_r_conj = wfc_r.conj()

        if self.use_gpu:
            nwfc, nx, ny, nz = wfc_r.shape
            if chunk_size == 0:
                # determine chunk_size based on free_mem
                chunk_size = int((free_gpu_memory / nwfc) ** (1 / 3) / 16 / 2)
                if chunk_size < 1:
                    logger.error(f"chunk_size smaller than one! Free memory: {free_gpu_memory / 10 ** 6:^g} MB")
                    raise RuntimeError

            for x in range(0, nx, chunk_size):
                for y in range(0, ny, chunk_size):
                    for z in range(0, nz, chunk_size):
                        # Define chunk slices
                        x_slice = slice(x, min(x + chunk_size, nx))
                        y_slice = slice(y, min(y + chunk_size, ny))
                        z_slice = slice(z, min(z + chunk_size, nz))

                        # Extract chunks
                        proj_chunk = projector[x_slice, y_slice, z_slice]
                        wfc_r_chunk = cp.asarray(wfc_r[:, x_slice, y_slice, z_slice])
                        wfc_r_conj_chunk = wfc_r_chunk.conj()

                        # Perform einsum on chunks
                        chunk_result = cp.einsum('ijk,aijk,bijk->ab', proj_chunk, wfc_r_conj_chunk, wfc_r_chunk)

                        # Aggregate results
                        self.overlap_matrix += chunk_result
            self.overlap_matrix = cp.maximum(self.overlap_matrix, self.overlap_matrix.T)
            self.localization, self.transformer = cp.linalg.eigh(self.overlap_matrix)
        else:
            self.overlap_matrix = np.einsum('ijk,aijk,bijk->ab', projector, wfc_r_conj, wfc_r)
            self.overlap_matrix = np.maximum(self.overlap_matrix, self.overlap_matrix.T)
            self.localization, self.transformer = np.linalg.eigh(self.overlap_matrix)

        localized_wfc_r = self.compute_localized_wfc(self.transformer, wfc_r)
        return self.localization, localized_wfc_r

    @staticmethod
    def compute_localized_wfc(transformer: np.ndarray, # [nwfc, nwfc]
                              wfc_r: np.ndarray):      # [nwfc, ...]
        wfc_r_shape = wfc_r.shape
        
        localized_wfc_r = None
        if isinstance(transformer, np.ndarray):
            localized_wfc_r = transformer.T @ wfc_r.reshape((wfc_r_shape[0], -1))
            localized_wfc_r = localized_wfc_r.reshape(wfc_r_shape)
        elif isinstance(transformer, cp.ndarray):
            # proceed in chunk
            chunk_size = 0
            nwfc, nx, ny, nz = wfc_r.shape
            # determine chunk_size based on free_mem
            chunk_size = int((free_gpu_memory / nwfc) ** (1 / 3) / 16)
            if chunk_size < 1:
                logger.error(f"chunk_size smaller than one! Free memory: {free_gpu_memory / 10 ** 6:^g} MB")
                raise RuntimeError

            localized_wfc_r = np.zeros_like(wfc_r)
            for x in range(0, nx, chunk_size):
                for y in range(0, ny, chunk_size):
                    for z in range(0, nz, chunk_size):
                        # Define chunk slices
                        x_slice = slice(x, min(x + chunk_size, nx))
                        y_slice = slice(y, min(y + chunk_size, ny))
                        z_slice = slice(z, min(z + chunk_size, nz))

                        # Extract chunks
                        wfc_r_chunk = cp.asarray(wfc_r[:, x_slice, y_slice, z_slice])
                        wfc_r_chunk_shape = wfc_r_chunk.shape
                        localized_wfc_r_chunk = transformer.T @ wfc_r_chunk.reshape((wfc_r_shape[0], -1))
                        localized_wfc_r_chunk = localized_wfc_r_chunk.reshape(wfc_r_chunk_shape).get()
                        localized_wfc_r[:, x_slice, y_slice, z_slice] = localized_wfc_r_chunk

        return localized_wfc_r
