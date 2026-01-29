import numpy as np
import bisect

class Material:
    def __init__(self, sigt, siga, sigs, nusigf):
        self.sigt = sigt
        self.siga = siga
        self.sigs = sigs
        self.nusigf = nusigf
    def getXs(self):
        return self.sigt, self.siga, self.sigs, self.nusigf

class Geo1D:
    def __init__(self, limits, mats):
        self.limits = limits
        self.mats = mats
    def getNbRegions(self):
        return len(self.limits) - 1
    def getXs(self, x):
        i = bisect.bisect_left(self.limits, x)
        mat = self.mats[i-1]
        return mat.getXs()


class Mesh1D:
    def __init__(self, nbs):
        nbs = [int(nb) for nb in nbs]
        self.nbs = nbs
    def getNb(self):
        aux = 1
        for nb in self.nbs:
            aux *= nb
        return aux
    def getX0s(self, X1s):
        xcoo = []
        for i in range(len(X1s)-1):
            xcoo.append(0.5*(X1s[i+1] + X1s[i]))
        return xcoo
    def getX1s(self, geo):
        xcoo = geo.limits
        xx = xcoo[0]
        xmesh = [xx]
        for igeo in range(geo.getNbRegions()):
            Dx = geo.limits[igeo+1] - geo.limits[igeo]
            for k in range(self.nbs[igeo]):
                hx = Dx / self.nbs[igeo]
                xx += hx
                xmesh.append(xx)
        return xmesh
    def getXs(self, x1s, x0s):
        xs = [x1s[0]]
        for i in range(len(x0s)):
            xs += [x0s[i], x1s[i+1]]
        return xs

class Scheme:
    def getName(self):
        return self.name
    pass

class Diamond(Scheme):
    def compute(self, psi, r1, r2):
        psi0 = (psi + r1) / (1 + r2)
        psi1 = 2*psi0 - psi
        return psi0, psi1
    
class DiamondFixup(Scheme):
    def compute(self, psi, r1, r2):
        psi0 = (psi + r1) / (1 + r2)
        psi1 = 2*psi0 - psi
        if psi1 < 0:
            psi1 = 0
        return psi0, psi1
    
class Step(Scheme):
    def compute(self, psi, r1, r2):
        psi0 = (psi + 2*r1) / (1 + 2*r2)
        psi1 = psi0
        return psi0, psi1

class SNSolver:
    def __init__(self, geo, mesh, source, sn, scheme=Diamond()):
        self.geo = geo
        self.mesh = mesh
        self.source = source
        self.sn = int(sn)
        self.scheme = scheme
    def computeError(self, x1, x2):
        aux1 = np.array(x1)
        aux2 = np.array(x2)
        err0 = np.max(np.abs(aux1 - aux2))
        return err0

    def solve(self, nbmax=100, eps=1.e-4):
        xs1 = self.mesh.getX1s(self.geo)
        xs0 = self.mesh.getX0s(xs1)
        xs = self.mesh.getXs(xs1, xs0)
        N = self.sn
        Nb = len(xs0)
        hxs = [xs1[i+1] - xs1[i] for i in range(Nb)]
        mu, w = np.polynomial.legendre.leggauss(N)
        phi_center = len(xs0)*[0] # flux at center
        sweep_toright = range(Nb)
        sweep_toleft = list(range(Nb-1, -1, -1))
        done = False
        for iter in range(nbmax): # inner iteration
            phi_center_new  = len(xs0)*[0]
            phi = len(xs)*[0] # flux at edges and centers
            for m in range(N): # loop over angles
                xmu = mu[m]
                if xmu > 0:
                    psi = 0
                    phi[0] += w[m]*psi
                    for i in range(Nb): # sweep in space
                        hx = hxs[i]
                        xi = xs0[i]
                        xs_tot, xs_abs, xs_scat, nu_xsfiss = self.geo.getXs(xi)
                        Q = xs_scat*phi_center[i]/2 + self.source(xi)/2 + nu_xsfiss*phi_center[i]/2
                        amu = xmu
                        r1 = hx*Q/amu/2
                        r2 = xs_tot*hx/amu/2
                        psi0, psi = self.scheme.compute(psi, r1, r2)
                        phi_center_new[i] += w[m]*psi0
                        phi[2*i+1] += w[m]*psi0
                        phi[2*i+2] += w[m]*psi
                        #print(xmu, xi, i, 2*i+1, 2*i+2, psi0, psi)
                else:
                    psi = 0
                    phi[-1] += w[m]*psi 
                    for i in range(Nb-1, -1, -1): # sweep in space
                        hx = hxs[i]
                        xi = xs0[i]
                        xs_tot, xs_abs, xs_scat, nu_xsfiss = self.geo.getXs(xi)
                        Q = xs_scat*phi_center[i]/2 + self.source(xi)/2 + nu_xsfiss*phi_center[i]/2
                        amu = abs(xmu)
                        r1 = hx*Q/amu/2
                        r2 = xs_tot*hx/amu/2
                        psi0, psi = self.scheme.compute(psi, r1, r2)
                        phi_center_new[i] += w[m]*psi0
                        phi[2*i+1] += w[m]*psi0
                        phi[2*i] += w[m]*psi
                        #print(xmu, xi, i, 2*i+1, 2*i, psi0, psi)
            error = self.computeError(phi_center_new, phi_center)
            phi_center = phi_center_new
            if error < eps:
                done = True
                break
        
        print("Iterations:", iter, "error:", error)
        self.flux_center = phi_center
        self.flux = phi
        self.mesh0 = xs0
        self.mesh = xs
        if not done:
            print("Warning: no convergence reached")
        return phi, xs


if __name__ == "__main__":
    mu, w = np.polynomial.legendre.leggauss(4)
    print(mu)
    print(w)
    mat1 = Material(1, 1, 0, 0)
    geo = Geo1D([0, 10], [mat1])
    mesh = Mesh1D([2])
    x1s = mesh.getX1s(geo)
    x0s = mesh.getX0s(x1s)
    xs = mesh.getXs(x1s, x0s)
    print(x1s, x0s, xs)
    def source(x):
        return 1.0
    solver = SNSolver(geo, mesh, source, sn=2, scheme=Step())
    flux, xs = solver.solve()
    print(xs)
    print(flux)