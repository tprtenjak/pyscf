#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# dia, para, make_h10, get_jk, make_h10giao, make_s10: Qiming Sun (pyscf-properties, nmr/rhf.py)
# AO density-matrix perturbation theory (DMPT) solvers: T. Prtenjak, L. A. Truflandier
#

'''
Non-relativistic RHF NMR shielding tensors from the first-order density matrix in the
(non-orthogonal) AO basis: density matrix perturbation theory (DMPT).

Self-contained: depends only on pyscf, numpy and scipy. It can be used as a stand-alone
module or placed in pyscf/prop/nmr/ next to rhf.py. It does not replace the standard
NMR class: it registers  mf.NMR_AODMPT().

    from pyscf import gto, scf
    import rhf_dmpt_ao                          # or: from pyscf.prop.nmr import rhf_dmpt_ao
    mf = scf.RHF(mol).run()
    sigma = rhf_dmpt_ao.NMR(mf, method='hpcp').kernel()      # (natm, 3, 3), ppm
    sigma = mf.NMR_AODMPT(method='tc2').kernel()

Conventions: D0 = C_occ C_occ^T (per spin, D0 S D0 = D0), D1 = dD/dB per spin (imaginary part,
real antisymmetric matrix), F1 = h1 + G[D1] with G[D1] = vresp(2 D1), vresp =
mf.gen_response(singlet=True, hermi=2) (J = 0 for an antisymmetric D1, G[D1] = -K[D1]).

First-order density (method):
  'sos'        McWeeny sum over states, AO projector form (needs the MO coefficients):
               D1 = -D0 S1 D0 + Y - Y^T,  Y = Cv y Co^T,  y_ai = (F1 - e_i S1)_ai / (e_i - e_a)
  'sylvester'  orbital-free: Sylvester equation  A Y + Y B = Q  for the first-order commutation
               condition, A = S^-1 F0 + a D0 S,  B = -F0 S^-1 + a (1 - S D0)  (shift a from the
               Gershgorin bounds makes A Y + Y B nonsingular); only the vir-occ projection
               (1 - D0 S) Y S D0 of the solution is used, which is exact
  'tc2'        orbital-free: first-order trace-correcting purification (Niklasson, Challacombe 2004)
  'hpcp'       orbital-free: first-order holes-particles canonical purification
               (Truflandier, Dianzinga, Bowler 2016; Palser, Manolopoulos 1998)
All four give the same D1 at convergence. The coupled equation D1 = L(h1 + G[D1]) + L_S(S1),
linear in D1, is solved with the block Krylov solver of PySCF CPHF (pyscf.lib.krylov,
Pople et al. 1979), the three field components together.
'''

import time
import numpy
import scipy.linalg
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import _vhf
from pyscf.scf import _response_functions  # noqa: F401  (registers mf.gen_response)
from pyscf.data import nist


# =============================================================================
# Integrals (pyscf-properties, nmr/rhf.py)
# =============================================================================
def dia(nmrobj, gauge_orig=None, shielding_nuc=None, dm0=None):
    '''Diamagnetic part of NMR shielding tensors (a.u.). See J. Olsen et al., TCA 90, 421 (1995).'''
    if shielding_nuc is None:
        shielding_nuc = nmrobj.shielding_nuc
    if dm0 is None:
        dm0 = nmrobj._scf.make_rdm1()
    mol = nmrobj.mol
    if gauge_orig is not None:
        mol.set_common_origin(gauge_orig)
    msc_dia = []
    for atm_id in shielding_nuc:
        with mol.with_rinv_origin(mol.atom_coord(atm_id)):
            if gauge_orig is None:
                h11 = mol.intor('int1e_giao_a11part', comp=9)
            else:
                h11 = mol.intor('int1e_cg_a11part', comp=9)
            e11 = numpy.einsum('xij,ij->x', h11, dm0).reshape(3, 3)
            e11 = e11 - numpy.eye(3) * e11.trace()
            if gauge_orig is None:
                h11 = mol.intor('int1e_a01gp', comp=9)
                e11 += numpy.einsum('xij,ij->x', h11, dm0).reshape(3, 3)
        msc_dia.append(e11)
    return numpy.array(msc_dia).reshape(-1, 3, 3)


def para_dm(nmrobj, dm1, shielding_nuc=None):
    '''Paramagnetic part of NMR shielding tensors (a.u.) from the AO first-order density matrix
    dm1 (3, nao, nao), per spin, antisymmetric:  sigma_para[x, y] = 2 Tr(D1_x h01_y).'''
    if shielding_nuc is None:
        shielding_nuc = nmrobj.shielding_nuc
    mol = nmrobj.mol
    msc_para = numpy.empty((len(shielding_nuc), 3, 3))
    for n, atm_id in enumerate(shielding_nuc):
        with mol.with_rinv_origin(mol.atom_coord(atm_id)):
            h01i = mol.intor_asymmetric('int1e_prinvxp', 3)          # = -Im[H^{01}]
        msc_para[n] = numpy.einsum('xij,yji->xy', dm1, h01i) * 2      # *2 for + c.c.
    return msc_para


def make_h10(mol, dm0, gauge_orig=None, verbose=logger.WARN):
    '''Imaginary part of the first-order Fock operator (bare, without G[D1]).'''
    log = logger.new_logger(mol, verbose)
    if gauge_orig is None:
        log.debug('First-order GIAO Fock matrix')
        h1 = -.5 * mol.intor('int1e_giao_irjxp', 3) + make_h10giao(mol, dm0)
    else:
        with mol.with_common_origin(gauge_orig):
            h1 = -.5 * mol.intor('int1e_cg_irxp', 3)
    return h1


def get_jk(mol, dm0):
    # J = Im[(i i|mu g nu) + (i gi|mu nu)] = -i (i i|mu g nu)
    # K = Im[(mu gi|i nu) + (mu i|i g nu)] = [-i (mu g i|i nu)] - h.c.
    intor = mol._add_suffix('int2e_ig1')
    vj, vk = _vhf.direct_mapdm(intor, 'a4ij', ('lk->s1ij', 'jk->s1il'), dm0, 3,
                               mol._atm, mol._bas, mol._env)
    vk = vk - numpy.swapaxes(vk, -1, -2)
    return -vj, -vk


def make_h10giao(mol, dm0):
    vj, vk = get_jk(mol, dm0)
    h1 = vj - .5 * vk
    h1 -= mol.intor_asymmetric('int1e_ignuc', 3)
    if mol.has_ecp():
        h1 -= mol.intor_asymmetric('ECPscalar_ignuc', 3)
    h1 -= mol.intor('int1e_igkin', 3)
    return h1


def make_s10(mol, gauge_orig=None):
    '''First-order overlap matrix (imaginary part) w.r.t. the external magnetic field.'''
    if gauge_orig is None:
        return -mol.intor_asymmetric('int1e_igovlp', 3)
    nao = mol.nao_nr()
    return numpy.zeros((3, nao, nao))


# =============================================================================
# AO-DMPT building blocks
# =============================================================================
def gershgorin_bounds(A):
    '''Lower and upper Gershgorin bounds of the spectrum of A (rows).'''
    d = numpy.diag(A)
    r = numpy.abs(A).sum(axis=1) - numpy.abs(d)
    return float((d - r).min()), float((d + r).max())


def commutator_rhs(F0, S0, D0, F1, S1, D1oo):
    '''First-order commutation residual  R = d/dB (F D S - S D F)  without the vir-occ part of D1:
    R = F1 D0 S - S D0 F1 + F0 D0 S1 - S1 D0 F0 + F0 D1oo S - S D1oo F0.'''
    return (F1 @ D0 @ S0 - S0 @ D0 @ F1 + F0 @ D0 @ S1 - S1 @ D0 @ F0
            + F0 @ D1oo @ S0 - S0 @ D1oo @ F0)


class _SOS:
    '''McWeeny sum over states in AO projector form.'''
    def __init__(self, nmrobj):
        mf = nmrobj._scf
        occ = mf.mo_occ > 0
        self.Co = mf.mo_coeff[:, occ]
        self.Cv = mf.mo_coeff[:, ~occ]
        self.eo = mf.mo_energy[occ]
        self.ev = mf.mo_energy[~occ]
        self.D0 = self.Co @ self.Co.T
        self.nsteps = 0

    def __call__(self, F1, S1):
        Co, Cv, D0 = self.Co, self.Cv, self.D0
        D1 = numpy.empty_like(F1)
        for x in range(F1.shape[0]):
            y = (Cv.T @ F1[x] @ Co - (Cv.T @ S1[x] @ Co) * self.eo) / (self.eo - self.ev[:, None])
            Y = Cv @ y @ Co.T
            D1[x] = -D0 @ S1[x] @ D0 + Y - Y.T
        return D1


class _Sylvester:
    '''Orbital-free first-order density from a nonsingular Sylvester equation.

    With Y = C y C^T (contravariant), (A Y + Y B) has the components
    (e_p - e_q + a [p occ] + a [q vir]) y_pq; with a = (e_max - e_min) + 1 (Gershgorin bounds of
    S^-1 F0) none of them vanishes. The vir-occ components, e_a - e_i, are not shifted, so
    Y_vo = (1 - D0 S) Y S D0 solves the first-order commutation condition
    F0 Y S - S Y F0 = -R exactly:  y_ai = (F1 - e_i S1)_ai / (e_i - e_a).
    '''
    def __init__(self, nmrobj, S0, Sinv, F0, D0):
        emin, emax = gershgorin_bounds(Sinv @ F0)
        a = (emax - emin) + 1.0
        M = S0.shape[0]
        I = numpy.eye(M)
        self.A = Sinv @ F0 + a * D0 @ S0
        self.B = -F0 @ Sinv + a * (I - S0 @ D0)
        self.Pv = I - D0 @ S0                   # (1 - D0 S): virtual projector (left)
        self.Po = S0 @ D0                       # S D0: occupied projector (right)
        self.S0, self.Sinv, self.F0, self.D0 = S0, Sinv, F0, D0
        self.nsteps = 0

    def __call__(self, F1, S1):
        D1 = numpy.empty_like(F1)
        for x in range(F1.shape[0]):
            D1oo = -self.D0 @ S1[x] @ self.D0
            Q = -self.Sinv @ commutator_rhs(self.F0, self.S0, self.D0, F1[x], S1[x], D1oo) @ self.Sinv
            Y = scipy.linalg.solve_sylvester(self.A, self.B, Q)
            Yvo = self.Pv @ Y @ self.Po
            D1[x] = D1oo + Yvo - Yvo.T
        return D1


class _Purification:
    '''Zeroth- and first-order density matrix purification in a non-orthogonal basis (TC2, HPCP).
    The zeroth-order trajectory (D0_n and the step coefficients) is computed once and stored;
    the first-order recursion is linear in (F1, S1) for a fixed number of steps.'''
    def __init__(self, method, S0, Sinv, F0, nocc, tol=1e-10, max_cycle=500, c_tol=1e-8):
        self.method = method
        self.S0, self.Sinv, self.F0, self.N = S0, Sinv, F0, nocc
        self.tol, self.max_cycle, self.c_tol = tol, max_cycle, c_tol
        M = S0.shape[0]
        self.emin, self.emax = gershgorin_bounds(Sinv @ F0)
        if method == 'tc2':
            D = ((self.emax * numpy.eye(M) - Sinv @ F0) @ Sinv) / (self.emax - self.emin)
        elif method == 'hpcp':
            theta = nocc / M
            self.mu = numpy.trace(Sinv @ F0) / M
            self.beta1 = theta
            self.beta2 = min(theta / (self.emax - self.mu), (1 - theta) / (self.mu - self.emin))
            D = self.beta1 * Sinv + self.beta2 * (self.mu * Sinv - Sinv @ F0 @ Sinv)
        else:
            raise ValueError(method)
        self.D = [D]                    # zeroth-order trajectory
        self.coef = []                  # alpha_n (TC2) or c_n (HPCP)
        self.converged0 = None          # step at which D0 converged
        self.nsteps = None              # number of first-order steps (set by the first call)

    def _zeroth_step(self):
        D, S = self.D[-1], self.S0
        ds = D @ S
        if self.method == 'tc2':
            alpha = 1.0 if self.N - numpy.trace(ds) > 0 else -1.0
            Dn = D + alpha * (D - ds @ D)
            self.coef.append(alpha)
        else:
            dsd = ds @ D
            dsdsd = dsd @ S @ D
            den = numpy.trace(ds - dsd @ S)
            # |den| = tr(DS - DSDS) ~ 0: D is idempotent to working precision. c = 1/2 turns the
            # HPCP step into McWeeny purification, 3 DSD - 2 DSDSD, whose first-order map projects
            # out the occ-occ and vir-vir blocks (c = 1 would keep them and flip the sign of vir-vir)
            c = numpy.trace((dsd - dsdsd) @ S) / den if abs(den) > self.c_tol else 0.5
            Dn = (1 - 2 * c) * D + 2 * (1 + c) * dsd - 2 * dsdsd
            self.coef.append(c)
        if self.converged0 is None and numpy.linalg.norm(Dn - D) < self.tol:
            self.converged0 = len(self.D)
        self.D.append(Dn)

    def _initial(self, F1, S1):
        Sinv, F0 = self.Sinv, self.F0
        sss = Sinv @ S1 @ Sinv
        if self.method == 'tc2':
            return (-self.emax * sss - Sinv @ (F1 - S1 @ Sinv @ F0 - F0 @ Sinv @ S1) @ Sinv) \
                / (self.emax - self.emin)
        return -self.beta1 * sss + self.beta2 * (-self.mu * sss - Sinv @ F1 @ Sinv
                                                 + sss @ F0 @ Sinv + Sinv @ F0 @ sss)

    def _first_step(self, n, D1, S1):
        D, S, k = self.D[n], self.S0, self.coef[n]
        ds = D @ S
        lin = D1 @ ds.T + D @ S1 @ D + ds @ D1               # d/dB (D S D)
        if self.method == 'tc2':
            return D1 + k * (D1 - lin)
        dsd = ds @ D
        cub = (D1 @ S @ dsd + D @ S1 @ dsd + ds @ D1 @ ds.T  # d/dB (D S D S D)
               + dsd @ S1 @ D + dsd @ S @ D1)
        return (1 - 2 * k) * D1 + 2 * (1 + k) * lin - 2 * cub

    def __call__(self, F1, S1):
        '''First-order density for (F1, S1), one field component at a time. The first call runs
        each component until D0 and D1 are converged and fixes the number of steps (the largest
        over the components); later calls (the coupled operator) use that number of steps, so
        that the map stays linear.'''
        adaptive = self.nsteps is None
        out = numpy.empty_like(F1)
        nmax = 0
        for x in range(F1.shape[0]):
            D1 = self._initial(F1[x], S1[x])
            n = 0
            while adaptive or n < self.nsteps:
                while len(self.D) <= n + 1:
                    self._zeroth_step()
                D1n = self._first_step(n, D1, S1[x])
                n += 1
                if adaptive:
                    err = numpy.linalg.norm(D1n - D1) / max(1.0, numpy.linalg.norm(D1n))
                    D1 = D1n
                    if (self.converged0 is not None and n > self.converged0 and err < self.tol) \
                            or n >= self.max_cycle:
                        break
                else:
                    D1 = D1n
            out[x] = D1
            nmax = max(nmax, n)
        if adaptive:
            self.nsteps = nmax
        return out


# =============================================================================
# First-order density matrix: uncoupled and coupled
# =============================================================================
def solve_dm1(nmrobj, method=None, h1=None, s1=None, with_cphf=None):
    '''First-order density matrix D1 (3, nao, nao), AO, per spin.'''
    if method is None:
        method = nmrobj.method
    if with_cphf is None:
        with_cphf = nmrobj.cphf
    mf = nmrobj._scf
    mol = nmrobj.mol
    log = logger.new_logger(nmrobj, nmrobj.verbose)
    t0 = (logger.process_clock(), logger.perf_counter())
    dm0 = mf.make_rdm1()
    if h1 is None:
        h1 = make_h10(mol, dm0, nmrobj.gauge_orig, log)
    if s1 is None:
        s1 = make_s10(mol, nmrobj.gauge_orig)
    t0 = log.timer('first-order integrals h1, s1', *t0)

    S0 = mf.get_ovlp()
    Sinv = numpy.linalg.inv(S0)
    F0 = mf.get_fock(dm=dm0)
    D0 = .5 * dm0
    nocc = int((mf.mo_occ > 0).sum())
    if method == 'sos':
        L = _SOS(nmrobj)
    elif method == 'sylvester':
        L = _Sylvester(nmrobj, S0, Sinv, F0, D0)
    elif method in ('tc2', 'hpcp'):
        L = _Purification(method, S0, Sinv, F0, nocc, tol=nmrobj.purif_tol, max_cycle=nmrobj.max_cycle_purif)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose from: 'sos', 'sylvester', 'tc2', 'hpcp'.")

    b = L(h1, s1)                                  # uncoupled D1
    t0 = log.timer(f'uncoupled D1 ({method})', *t0)
    if not with_cphf:
        return b

    nset, M = h1.shape[0], h1.shape[1]
    vresp = mf.gen_response(singlet=True, hermi=2)
    zero = numpy.zeros((nset, M, M))
    count = {'nresp': 0}

    def aop(x):
        x = numpy.asarray(x).reshape(-1, M, M)
        count['nresp'] += x.shape[0]
        G = vresp(2 * x)                               # G[D1] per spin
        return -L(G, zero[:x.shape[0]]).reshape(x.shape[0], -1)

    D1 = lib.krylov(aop, b.reshape(nset, -1), tol=nmrobj.conv_tol, max_cycle=nmrobj.max_cycle_cphf, lindep=nmrobj.krylov_lindep,
                    verbose=log)
    D1 = numpy.asarray(D1).reshape(nset, M, M)
    nmrobj.nresp = count['nresp']
    nmrobj.purif_steps = getattr(L, 'nsteps', 0)
    log.timer(f'coupled D1 ({method}, {count["nresp"]} response builds)', *t0)
    return D1


# =============================================================================
class NMR(lib.StreamObject):
    '''RHF NMR shielding tensors from AO density matrix perturbation theory.

    Attributes:
        method : 'hpcp' (default), 'tc2', 'sylvester' or 'sos'
        cphf : bool, coupled (True) or uncoupled first-order density
        conv_tol, max_cycle_cphf : block Krylov solver of the coupled equation
        krylov_lindep : linear-dependence threshold of pyscf.lib.krylov (default 1e-13, as in PySCF
                        CPHF; it limits the attainable accuracy, smaller values converge further)
        purif_tol, max_cycle_purif : TC2 / HPCP purification
        gauge_orig : None for GIAO, or a common gauge origin
    '''
    def __init__(self, scf_method, method='hpcp'):
        self.mol = scf_method.mol
        self.verbose = scf_method.mol.verbose
        self.stdout = scf_method.mol.stdout
        self.chkfile = scf_method.chkfile
        self._scf = scf_method
        self.shielding_nuc = range(self.mol.natm)
        self.gauge_orig = None
        self.method = method
        self.cphf = True
        self.max_cycle_cphf = 50
        self.conv_tol = 1e-9
        self.krylov_lindep = 1e-13         # pyscf.lib.krylov default (DSOLVE_LINDEP)
        self.purif_tol = 1e-10
        self.max_cycle_purif = 500
        self.dm10 = None
        self.nresp = None
        self.purif_steps = None
        self._keys = set(self.__dict__.keys())

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('\n******** %s for %s ********', self.__class__, self._scf.__class__)
        log.info('gauge = %s', 'GIAO' if self.gauge_orig is None else 'common gauge %s' % str(self.gauge_orig))
        log.info('first-order density: AO-DMPT, method = %s, coupled = %s', self.method, self.cphf)
        log.info('Krylov conv_tol = %g, max_cycle = %d', self.conv_tol, self.max_cycle_cphf)
        if not self._scf.converged:
            log.warn('Ground state SCF is not converged')
        return self

    def kernel(self, dm1=None):
        return self.shielding(dm1)

    def shielding(self, dm1=None):
        cput0 = (logger.process_clock(), logger.perf_counter())
        self.check_sanity()
        self.dump_flags()
        unit_ppm = nist.ALPHA**2 * 1e6
        msc_dia = self.dia(self.gauge_orig)
        if dm1 is None:
            self.dm10 = dm1 = self.solve_dm1()
        msc_para = self.para_dm(dm1)
        e11 = (msc_para + msc_dia) * unit_ppm
        logger.timer(self, 'NMR shielding (AO-DMPT)', *cput0)
        if self.verbose >= logger.NOTE:
            for i, atm_id in enumerate(self.shielding_nuc):
                _write(self.stdout, e11[i], '\ntotal shielding of atom %d %s' % (atm_id, self.mol.atom_symbol(atm_id)))
                _write(self.stdout, msc_dia[i] * unit_ppm, 'dia-magnetic contribution')
                _write(self.stdout, msc_para[i] * unit_ppm, 'para-magnetic contribution')
        return e11

    dia = dia
    para_dm = para_dm
    solve_dm1 = solve_dm1

    def get_fock(self, dm0=None, gauge_orig=None):
        if dm0 is None:
            dm0 = self._scf.make_rdm1()
        return make_h10(self.mol, dm0, self.gauge_orig if gauge_orig is None else gauge_orig)

    def get_ovlp(self, mol=None, gauge_orig=None):
        return make_s10(mol or self.mol, self.gauge_orig if gauge_orig is None else gauge_orig)


RHF = NMR

from pyscf import scf  # noqa: E402
scf.hf.RHF.NMR_AODMPT = lib.class_as_method(NMR)


def _write(stdout, msc3x3, title):
    stdout.write('%s\n' % title)
    stdout.write('B_x %s\n' % str(msc3x3[0]))
    stdout.write('B_y %s\n' % str(msc3x3[1]))
    stdout.write('B_z %s\n' % str(msc3x3[2]))
    stdout.flush()


if __name__ == '__main__':
    from pyscf import gto
    mol = gto.M(atom='O 0 0 0; H 0 .757 .587; H 0 -.757 .587', basis='6-31g*', verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    ref = None
    for m in ('sos', 'sylvester', 'tc2', 'hpcp'):
        t = time.perf_counter()
        nmr = NMR(mf, method=m)
        s = nmr.kernel()
        iso = [numpy.trace(x) / 3 for x in s]
        if ref is None:
            ref = s
        print(f'{m:10s} sigma_iso O {iso[0]:.6f}  H {iso[1]:.6f}  max|dsigma| vs sos {abs(s - ref).max():.1e} ppm  '
              f'builds {nmr.nresp}  purification steps {nmr.purif_steps}  {time.perf_counter() - t:.2f} s')
