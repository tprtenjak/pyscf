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
# MO density-matrix perturbation theory (MO-DMPT), density fitting: T. Prtenjak, L. A. Truflandier
#

'''
Non-relativistic RHF NMR shielding tensors from MO density matrix perturbation theory (MO-DMPT),
optionally with density fitting of both two-electron terms.

Self-contained: depends only on pyscf, numpy and scipy. It can be used as a stand-alone
module or placed in pyscf/prop/nmr/ next to rhf.py. It does not replace the standard
NMR class: it registers  mf.NMR_MODMPT().

    from pyscf import gto, scf
    import rhf_dmpt_mo                          # or: from pyscf.prop.nmr import rhf_dmpt_mo
    mf = scf.RHF(mol).run()
    sigma = rhf_dmpt_mo.NMR(mf).kernel()                        # exact two-electron integrals
    sigma = rhf_dmpt_mo.NMR(mf).density_fit().kernel()          # DF h10 + DF response
    sigma = mf.NMR_MODMPT(method='tc2').density_fit(auxbasis='def2-universal-jkfit').kernel()

MO-DMPT. In the canonical MO basis of the converged SCF (S = 1, F0 = diag(e)), every zeroth-order
purification iterate is diagonal, d_n(p), and the first-order recursion acts on each matrix
element (p, q) separately:
    TC2   X' = X [1 + a_n (1 - d_p - d_q)]                                   - a_n d_p d_q S1
    HPCP  X' = X [(1 - 2c_n) + 2(1 + c_n)(d_p + d_q) - 2(d_p^2 + d_p d_q + d_q^2)]
               + d_p d_q [2(1 + c_n) - 2(d_p + d_q)] S1
For S1 = 0 the purification of F1 is a Hadamard product, D1 = kappa o F1, with a kernel kappa
computed once from the zeroth-order trajectory: O(n M^2) once, O(M^2) per application.
Coupled equation: D1 - kappa o G[D1] = b (b = purified h1, S1). kappa_vo -> 1/(e_i - e_a) and
kappa_oo, kappa_vv -> 0 for a converged trajectory, so only the vir-occ block x is iterated;
the occ-occ block (GIAO, -S1_oo) is carried in the Krylov vector z = (x, y_oo) as in
pyscf.scf.cphf.solve_withs1. Block Krylov solver: pyscf.lib.krylov (Pople et al. 1979).

Density fitting (density_fit()), field-independent auxiliary Gaussians, Coulomb metric:
  response   G[D1] = -K[D1], D1_AO = L R^T (rank 2 N_occ), L = [Cv x + Co y, -Co], R = [Co, Cv x],
             K = sum_P (B^P L)(B^P R)^T, cost O(naux M^2 N_occ) per density
  h10        G^(1)[D0] with (ij|kl)^B ~ sum_PQ (ij|P)^B [V^-1]_PQ (Q|kl), (ij|P)^B = int3c2e_ig1
             (Loibl, Manby, Schuetz, Mol. Phys. 108, 477 (2010)); one-electron terms exact.
'''

import time
import numpy
import scipy.linalg
from pyscf import lib, gto, df
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


def _h10_one_electron(mol):
    h1 = -mol.intor_asymmetric('int1e_ignuc', 3)
    if mol.has_ecp():
        h1 -= mol.intor_asymmetric('ECPscalar_ignuc', 3)
    h1 -= mol.intor('int1e_igkin', 3)
    return h1


def make_h10giao(mol, dm0):
    vj, vk = get_jk(mol, dm0)
    return vj - .5 * vk + _h10_one_electron(mol)


def make_s10(mol, gauge_orig=None):
    '''First-order overlap matrix (imaginary part) w.r.t. the external magnetic field.'''
    if gauge_orig is None:
        return -mol.intor_asymmetric('int1e_igovlp', 3)
    nao = mol.nao_nr()
    return numpy.zeros((3, nao, nao))


# =============================================================================
# Density fitting: GIAO h10 (two-electron part) and factorised response exchange
# =============================================================================
def _int3c(mol, auxmol, intor, comp, ish0, ish1):
    '''(ij|P), i in shells [ish0, ish1), all j, all auxiliary shells: (comp, ni, nao, naux).'''
    atm, bas, env = gto.mole.conc_env(mol._atm, mol._bas, mol._env, auxmol._atm, auxmol._bas, auxmol._env)
    sl = (ish0, ish1, 0, mol.nbas, mol.nbas, mol.nbas + auxmol.nbas)
    out = gto.moleintor.getints(mol._add_suffix(intor), atm, bas, env, sl, comp, 0, 's1')
    ao_loc = mol.ao_loc_nr()
    return numpy.asarray(out).reshape(comp, ao_loc[ish1] - ao_loc[ish0], mol.nao_nr(), auxmol.nao_nr())


def _metric_solver(V, lindep=1e-12):
    try:
        cf = scipy.linalg.cho_factor(V, lower=True)
        return lambda X: scipy.linalg.cho_solve(cf, X)
    except numpy.linalg.LinAlgError:
        w, U = numpy.linalg.eigh(V)
        keep = w > lindep * w.max()
        Ui = U[:, keep] / w[keep]
        return lambda X: Ui @ (U[:, keep].T @ X)


def _shell_blocks(mol, max_rows):
    ao_loc = mol.ao_loc_nr()
    blocks, i0 = [], 0
    for i in range(mol.nbas):
        if ao_loc[i + 1] - ao_loc[i0] > max_rows and i > i0:
            blocks.append((i0, i))
            i0 = i
    blocks.append((i0, mol.nbas))
    return [(a, b, ao_loc[a], ao_loc[b]) for a, b in blocks]


def get_jk_df(mol, auxmol, mo_occ_coeff, occ, max_memory=4000):
    '''DF analogue of get_jk(mol, P), P = sum_n occ_n c_n c_n^T:
        c = V^-1 sum_kl (Q|kl) P_lk,            vj_ij = sum_Q (ij|Q)^B c_Q
        Z = V^-1 R, R_{Q,n,l} = sum_k (Q|lk) Cp_kn,  vk_il = sum_{Q,n} [sum_j (ij|Q)^B Cp_jn] Z_{Q,n,l}
    The field-derivative integrals are used once, AO row block by row block.'''
    Cp = mo_occ_coeff * numpy.sqrt(occ)[None, :]
    P = Cp @ Cp.T
    M, N = Cp.shape
    naux = auxmol.nao_nr()
    solve = _metric_solver(auxmol.intor('int2c2e'))
    rows = max(1, int(max_memory * 1e6 / 10 / (3 * M * naux * 8)))
    blocks = _shell_blocks(mol, rows)
    R = numpy.empty((naux, N, M))
    craw = numpy.zeros(naux)
    for a, b, p0, p1 in blocks:
        X = _int3c(mol, auxmol, 'int3c2e', 1, a, b)[0]                      # (ni, M, naux)
        R[:, :, p0:p1] = numpy.einsum('lkQ,kn->Qnl', X, Cp, optimize=True)
        craw += numpy.einsum('lkQ,lk->Q', X, P[p0:p1], optimize=True)
    Z = solve(R.reshape(naux, -1)).reshape(naux * N, M)
    del R
    c = solve(craw)
    vj = numpy.empty((3, M, M))
    vk = numpy.empty((3, M, M))
    for a, b, p0, p1 in blocks:
        G = _int3c(mol, auxmol, 'int3c2e_ig1', 3, a, b)                     # (3, ni, M, naux)
        vj[:, p0:p1] = G @ c
        T = numpy.einsum('xijQ,jn->xiQn', G, Cp, optimize=True).reshape(3 * (p1 - p0), naux * N)
        vk[:, p0:p1] = (T @ Z).reshape(3, p1 - p0, M)
    vk = vk - vk.transpose(0, 2, 1)
    return -vj, -vk


def make_h10_df(mol, auxmol, mo_occ_coeff, occ, max_memory=4000):
    '''GIAO h10 with the two-electron term G^(1)[D0] density fitted; one-electron terms exact.'''
    vj, vk = get_jk_df(mol, auxmol, mo_occ_coeff, occ, max_memory)
    return -.5 * mol.intor('int1e_giao_irjxp', 3) + vj - .5 * vk + _h10_one_electron(mol)


def k_factorized(with_df, L, R):
    '''K[L R^T] = sum_P (B^P L)(B^P R)^T with the Cholesky-decomposed 3-index tensor of with_df;
    L, R: (n, nao, k). Returns (n, nao, nao).'''
    n, M, k = L.shape
    K = numpy.zeros((n, M, M))
    Lc = numpy.ascontiguousarray(L.transpose(1, 0, 2).reshape(M, n * k))
    Rc = numpy.ascontiguousarray(R.transpose(1, 0, 2).reshape(M, n * k))
    for blk in with_df.loop():
        B = lib.unpack_tril(blk)
        nb = B.shape[0]
        BL = (B.reshape(nb * M, M) @ Lc).reshape(nb, M, n, k)
        BR = (B.reshape(nb * M, M) @ Rc).reshape(nb, M, n, k)
        for c in range(n):
            X = numpy.ascontiguousarray(BL[:, :, c, :].transpose(1, 0, 2)).reshape(M, nb * k)
            Y = numpy.ascontiguousarray(BR[:, :, c, :].transpose(1, 0, 2)).reshape(M, nb * k)
            K[c] += X @ Y.T
    return K


# =============================================================================
# MO-DMPT: elementwise purification, kernel, coupled solve
# =============================================================================
class _ElementwisePurification:
    '''Zeroth-order purification on the orbital energies (diagonal in the canonical MO basis) and
    the elementwise first-order recursion. kernel(): kappa (M x M); first_order(F1, S1): purified
    first-order density for MO matrices F1, S1 (n, M, M), same number of steps as the kernel.'''
    def __init__(self, method, mo_energy, nocc, tol=1e-10, max_cycle=500, c_tol=1e-8):
        e = numpy.asarray(mo_energy, dtype=float)
        M = e.size
        self.method, self.e, self.N = method, e, nocc
        self.tol, self.max_cycle, self.c_tol = tol, max_cycle, c_tol
        self.emin, self.emax = float(e.min()), float(e.max())
        if method == 'tc2':
            d = (self.emax - e) / (self.emax - self.emin)
            self.k0 = -1.0 / (self.emax - self.emin)
        elif method == 'hpcp':
            theta = nocc / M
            self.mu = e.mean()
            self.beta1 = theta
            self.beta2 = min(theta / (self.emax - self.mu), (1 - theta) / (self.mu - self.emin))
            d = self.beta1 + self.beta2 * (self.mu - e)
            self.k0 = -self.beta2
        else:
            raise ValueError(f"Unknown method '{method}'. Choose from: 'tc2', 'hpcp'.")
        self.d = [d]
        self.coef = []
        self.kappa, self.nsteps = self._kernel()

    def _zeroth_step(self):
        d = self.d[-1]
        if self.method == 'tc2':
            a = 1.0 if self.N - d.sum() > 0 else -1.0
            dn = d + a * (d - d * d)
        else:
            den = (d - d * d).sum()
            # den ~ 0: idempotent to working precision; c = 1/2 is McWeeny purification
            a = ((d * d - d ** 3).sum() / den) if abs(den) > self.c_tol else 0.5
            dn = (1 - 2 * a) * d + 2 * (1 + a) * d * d - 2 * d ** 3
        self.coef.append(a)
        self.d.append(dn)
        return float(numpy.abs(dn - d).max())

    def _gh(self, n):
        dp = self.d[n][:, None]
        dq = self.d[n][None, :]
        a = self.coef[n]
        if self.method == 'tc2':
            return 1 + a * (1 - dp - dq), -a * dp * dq
        g = (1 - 2 * a) + 2 * (1 + a) * (dp + dq) - 2 * (dp * dp + dp * dq + dq * dq)
        h = dp * dq * (2 * (1 + a) - 2 * (dp + dq))
        return g, h

    def _kernel(self):
        M = self.e.size
        k = numpy.full((M, M), self.k0)
        conv0 = None
        for n in range(self.max_cycle):
            err0 = self._zeroth_step()
            if conv0 is None and err0 < self.tol:
                conv0 = n
            g, _ = self._gh(n)
            kn = g * k
            err1 = numpy.abs(kn - k).max() / max(1.0, numpy.abs(kn).max())
            k = kn
            if conv0 is not None and err1 < self.tol:
                return k, n + 1
        return k, self.max_cycle

    def first_order(self, F1, S1):
        e = self.e
        es = S1 * e[None, None, :] + e[None, :, None] * S1          # S1 F0 + F0 S1 (MO)
        if self.method == 'tc2':
            X = (-self.emax * S1 - (F1 - es)) / (self.emax - self.emin)
        else:
            X = -self.beta1 * S1 + self.beta2 * (-self.mu * S1 - F1 + es)
        for n in range(self.nsteps):
            g, h = self._gh(n)
            X = g * X + h * S1
        return X


def solve_dm1(nmrobj, method=None, h1=None, s1=None, with_cphf=None):
    '''First-order density matrix (3, nao, nao), AO, per spin, from MO-DMPT.'''
    if method is None:
        method = nmrobj.method
    if with_cphf is None:
        with_cphf = nmrobj.cphf
    mf, mol = nmrobj._scf, nmrobj.mol
    log = logger.new_logger(nmrobj, nmrobj.verbose)
    t0 = (logger.process_clock(), logger.perf_counter())
    C, e, occ = mf.mo_coeff, mf.mo_energy, mf.mo_occ
    o = occ > 0
    N = int(o.sum())
    Co, Cv = C[:, o], C[:, ~o]
    dm0 = mf.make_rdm1()
    with_df = nmrobj.with_df
    if with_df is not None and with_df._cderi is None:
        with_df.build()
        t0 = log.timer('DF 3-index tensor', *t0)
    if h1 is None:
        if nmrobj.gauge_orig is None and with_df is not None and nmrobj.df_h10:
            h1 = make_h10_df(mol, with_df.auxmol, Co, occ[o], max_memory=nmrobj.max_memory)
            t0 = log.timer('first-order GIAO Fock matrix (DF)', *t0)
        else:
            h1 = make_h10(mol, dm0, nmrobj.gauge_orig, log)
            t0 = log.timer('first-order Fock matrix', *t0)
    if s1 is None:
        s1 = make_s10(mol, nmrobj.gauge_orig)

    P = _ElementwisePurification(method, e, N, tol=nmrobj.purif_tol, max_cycle=nmrobj.max_cycle_purif)
    kappa = P.kappa
    nmrobj.purif_steps = P.nsteps
    nmrobj.kappa_Q = max(float(numpy.abs(kappa[:N, :N]).max()), float(numpy.abs(kappa[N:, N:]).max()))
    h1_mo = lib.einsum('pi,xpq,qj->xij', C, h1, C)
    s1_mo = lib.einsum('pi,xpq,qj->xij', C, s1, C)
    b = P.first_order(h1_mo, s1_mo)                       # uncoupled D1 (MO)
    t0 = log.timer(f'MO-DMPT kernel and uncoupled D1 ({method}, {P.nsteps} steps, '
                   f'max|kappa_oo,vv| = {nmrobj.kappa_Q:.1e})', *t0)

    if with_cphf:
        nset = b.shape[0]
        Nv = C.shape[1] - N
        nvo = Nv * N
        k_vo = kappa[N:, :N]
        vresp = None if with_df is not None else mf.gen_response(singlet=True, hermi=2)
        count = {'nresp': 0}

        def G_vo(x, y):
            '''vir-occ block of G[D1] for D1_MO = (vo: x, ov: -x^T, oo: y).'''
            W = numpy.einsum('pa,xai->xpi', Cv, x, optimize=True)
            if with_df is not None:
                Cb = numpy.broadcast_to(Co, W.shape)
                L = numpy.concatenate([W + numpy.einsum('pj,xji->xpi', Co, y, optimize=True), -Cb], axis=2)
                R = numpy.concatenate([Cb, W], axis=2)
                V = -k_factorized(with_df, L, R)                 # G = -K for antisymmetric D1
            else:
                Y = numpy.einsum('xpi,qi->xpq', W, Co, optimize=True)
                Dao = Y - Y.transpose(0, 2, 1) + numpy.einsum('pi,xij,qj->xpq', Co, y, Co, optimize=True)
                V = vresp(2 * Dao)
            count['nresp'] += x.shape[0]
            return numpy.einsum('pa,xpq,qi->xai', Cv, V, Co, optimize=True)

        def aop(z):
            z = numpy.asarray(z).reshape(-1, nvo + N * N)
            out = numpy.zeros_like(z)
            out[:, :nvo] = (-(k_vo * G_vo(z[:, :nvo].reshape(-1, Nv, N),
                                          z[:, nvo:].reshape(-1, N, N)))).reshape(z.shape[0], -1)
            return out

        bz = numpy.concatenate([b[:, N:, :N].reshape(nset, -1), b[:, :N, :N].reshape(nset, -1)], axis=1)
        z = numpy.asarray(lib.krylov(aop, bz, tol=nmrobj.conv_tol, max_cycle=nmrobj.max_cycle_cphf, lindep=nmrobj.krylov_lindep,
                                     verbose=log)).reshape(nset, -1)
        x = z[:, :nvo].reshape(nset, Nv, N)
        b[:, N:, :N] = x
        b[:, :N, N:] = -x.transpose(0, 2, 1)
        nmrobj.nresp = count['nresp']
        log.timer(f'coupled MO-DMPT ({count["nresp"]} response builds)', *t0)
    return lib.einsum('pi,xij,qj->xpq', C, b, C)


# =============================================================================
class NMR(lib.StreamObject):
    '''RHF NMR shielding tensors from MO density matrix perturbation theory.

    Attributes:
        method : 'hpcp' (default) or 'tc2' zeroth- and first-order purification
        cphf : bool, coupled (True) or uncoupled first-order density
        conv_tol, max_cycle_cphf : block Krylov solver of the coupled equation
        krylov_lindep : linear-dependence threshold of pyscf.lib.krylov (default 1e-13, as in PySCF
                        CPHF; it limits the attainable accuracy, smaller values converge further)
        purif_tol, max_cycle_purif : purification (kernel) convergence
        with_df : None (exact two-electron integrals) or a pyscf.df.DF object; set by density_fit()
        df_h10 : with density fitting, also fit the GIAO two-electron term of h10 (default True)
        gauge_orig : None for GIAO, or a common gauge origin
    '''
    def __init__(self, scf_method, method='hpcp'):
        self.mol = scf_method.mol
        self.verbose = scf_method.mol.verbose
        self.stdout = scf_method.mol.stdout
        self.chkfile = scf_method.chkfile
        self.max_memory = scf_method.max_memory
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
        self.with_df = None
        self.df_h10 = True
        self.dm10 = None
        self.nresp = None
        self.purif_steps = None
        self.kappa_Q = None
        self._keys = set(self.__dict__.keys())

    def density_fit(self, auxbasis=None, h10=True):
        '''Density fitting of the response exchange and (h10=True) of the GIAO h10 two-electron
        term. auxbasis: as for mf.density_fit(); None takes PySCF's default (*-jkfit).'''
        self.with_df = df.DF(self.mol, auxbasis=auxbasis)
        self.with_df.max_memory = self.max_memory
        self.df_h10 = h10
        return self

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('\n******** %s for %s ********', self.__class__, self._scf.__class__)
        log.info('gauge = %s', 'GIAO' if self.gauge_orig is None else 'common gauge %s' % str(self.gauge_orig))
        log.info('first-order density: MO-DMPT, method = %s, coupled = %s', self.method, self.cphf)
        log.info('density fitting: %s', 'no' if self.with_df is None else
                 f'response K{" and h10" if self.df_h10 else ""}, auxbasis = {self.with_df.auxbasis or "PySCF default (jkfit)"}')
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
        logger.timer(self, 'NMR shielding (MO-DMPT)', *cput0)
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
scf.hf.RHF.NMR_MODMPT = lib.class_as_method(NMR)


def _write(stdout, msc3x3, title):
    stdout.write('%s\n' % title)
    stdout.write('B_x %s\n' % str(msc3x3[0]))
    stdout.write('B_y %s\n' % str(msc3x3[1]))
    stdout.write('B_z %s\n' % str(msc3x3[2]))
    stdout.flush()


if __name__ == '__main__':
    mol = gto.M(atom='O 0 0 0; H 0 .757 .587; H 0 -.757 .587', basis='def2-tzvp', verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    ref = None
    for label, make in [('hpcp exact', lambda: NMR(mf)), ('tc2 exact', lambda: NMR(mf, method='tc2')),
                        ('hpcp DF', lambda: NMR(mf).density_fit()),
                        ('hpcp DF response only', lambda: NMR(mf).density_fit(h10=False))]:
        t = time.perf_counter()
        nmr = make()
        s = nmr.kernel()
        if ref is None:
            ref = s
        iso = [numpy.trace(x) / 3 for x in s]
        print(f'{label:22s} sigma_iso O {iso[0]:.6f}  H {iso[1]:.6f}  max|dsigma| vs hpcp exact '
              f'{abs(s - ref).max():.1e} ppm  builds {nmr.nresp}  steps {nmr.purif_steps}  '
              f'max|kappa_oo,vv| {nmr.kappa_Q:.0e}  {time.perf_counter() - t:.2f} s')
