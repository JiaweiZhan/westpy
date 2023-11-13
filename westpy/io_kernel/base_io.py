class base_io:
    def __init__(self):
        self.ik = 0
        self.xk = [0.0, 0.0, 0.0]
        self.ispin = 0
        self.gamma_only = True
        self.scalef = 0.0
        self.ngw = 0
        self.igwx = 0
        self.npol = 0
        self.nbnd = 0
        self.b1 = [0.0, 0.0, 0.0]
        self.b2 = [0.0, 0.0, 0.0]
        self.b3 = [0.0, 0.0, 0.0]
        self.mill = None
        self.evc = None
        self.fftw = None
        self.pdepg = None
        self.pdepeig = None
