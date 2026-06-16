"""
frc_hybrid_pic.py
=================================================================
Hybrid-kinetic PIC (kinetic ions + massless fluid electrons)
Field-Reversed Configuration (FRC) simulation built on WarpX/PICMI.

CHANGELOG (this revision -- incorporates 9 critical physics/numeric fixes):
  [FIX-1]  Merged scenarios now correctly use a 2D analytic flux function
           (racetrack closure via z^4) to establish a true separatrix.
  [FIX-2]  Bernstein probe cadence is decoupled from spatial snapshots
           (PROBE_EVERY=5) extending the Nyquist limit to ~12.6 w_ci.
  [FIX-3]  Fusion registration parameterized and cross-lobe collisions
           explicitly added for the merging scenario.
  [FIX-4]  3D grid cell counts now scale by 2*nr across [-L_r, L_r].
  [FIX-5]  Species definitions stripped of conflicting particle_type/charge_state;
           strictly reliant on numeric Q_E and M_D.
  [FIX-6]  Added _get_first() wrapper fallback to prevent silent failures on Er.
  [FIX-7]  Energies properly weighted (2*pi*r dV for B-field, macro-weights for ions).
  [FIX-8]  T_i extracted exclusively from var(u_z) to avoid u_phi shear inflation.
  [FIX-9a] _field_video valid-frame zipping synchronized.
  [FIX-9b] Poloidal flux comment corrected to left Riemann sum.
  [FIX-9c] Full radial line-out of Er stored for future w-k 2D FFTs.
  [FIX-9e] Precedence trap parenthesized in register_dd_fusion.
"""

from __future__ import annotations
import argparse
import numpy as np

# ----------------------------------------------------------------------------
# Fundamental constants (SI)
# ----------------------------------------------------------------------------
Q_E = 1.602176634e-19
M_E = 9.1093837015e-31
U_AMU = 1.66053906660e-27
M_D = 2.013553212745 * U_AMU  # deuteron (D nucleus)
M_HE3 = 3.014932247175 * U_AMU
M_T = 3.015500716210 * U_AMU
M_P = 1.007276466621 * U_AMU
M_N = 1.008664915950 * U_AMU
EPS0 = 8.8541878128e-12
MU0 = 1.25663706212e-6
C0 = 299792458.0


# ============================================================================
# CONFIGURATION
# ============================================================================
class CFG:
    GEOMETRY = "RZ"
    SCENARIO = "merging"

    N_NULL = 1.0e22
    T_I = 2000.0
    T_E = 500.0
    R_SEP = 0.020
    ELONGATION = 4.0
    FIELD_SHARP = 0.5

    B_EXT = None
    R_NULL = None

    ALPHA_HALL = 1.0
    E_GAMMA = 5.0 / 3.0
    ETA = 1.0e-4
    ETA_HYPER = 1.0e-10

    CELLS_PER_DI = 4
    PPC = 200
    DT_WCI = 0.05
    SUBSTEPS = 250
    N_BG_FRAC = 0.05

    MAX_STEPS = 20000
    DIAG_EVERY = 50
    PROBE_EVERY = 5  # Nyquist limit ~ 12.6 w_ci

    VIDEO = True
    VIDEO_FPS = 12
    VIDEO_DPI = 110
    VIDEO_MAX_FRAMES = 400
    FIELD_DOWNSAMPLE = 1

    MERGE_OFFSET = None
    MERGE_MACH = 0.5
    COMPRESS_RATIO = 2.0
    COMPRESS_TAU = None

    DO_COULOMB = True
    DO_FUSION = True
    FUSION_MULT = 1.0e6

    TILT_PERT = 0.02


# ============================================================================
# PARAMETER DERIVATION
# ============================================================================
def derive_parameters(cfg: CFG = CFG, verbose: bool = True) -> dict:
    n = cfg.N_NULL
    Ti = cfg.T_I
    Te = cfg.T_E
    mi = M_D

    p_null = n * (Ti + Te) * Q_E
    B_e = np.sqrt(2.0 * MU0 * p_null)
    cfg.B_EXT = B_e
    cfg.R_NULL = cfg.R_SEP / np.sqrt(2.0)

    w_pe = np.sqrt(n * Q_E ** 2 / (EPS0 * M_E))
    w_pi = np.sqrt(n * Q_E ** 2 / (EPS0 * mi))
    w_ce = Q_E * B_e / M_E
    w_ci = Q_E * B_e / mi

    v_the = np.sqrt(Te * Q_E / M_E)
    v_thi = np.sqrt(Ti * Q_E / mi)

    lam_D = v_the / w_pe
    rho_e = v_the / w_ce
    rho_i = v_thi / w_ci
    d_e = C0 / w_pe
    d_i = C0 / w_pi

    v_A = B_e / np.sqrt(MU0 * n * mi)
    beta = p_null / (B_e ** 2 / (2.0 * MU0))
    s_bar = (cfg.R_SEP - cfg.R_NULL) / rho_i

    dx = min(d_i, rho_i) / cfg.CELLS_PER_DI
    dt = cfg.DT_WCI / w_ci

    cfg.MERGE_OFFSET = 1.5 * cfg.ELONGATION * cfg.R_SEP
    cfg.COMPRESS_TAU = 20.0 * (2 * np.pi / w_ci)

    out = dict(p_null=p_null, B_e=B_e, w_pe=w_pe, w_pi=w_pi, w_ce=w_ce, w_ci=w_ci,
               v_the=v_the, v_thi=v_thi, lam_D=lam_D, rho_e=rho_e, rho_i=rho_i,
               d_e=d_e, d_i=d_i, v_A=v_A, beta=beta, s_bar=s_bar, dx=dx, dt=dt,
               T_ci=2 * np.pi / w_ci)

    if verbose:
        print("=" * 70)
        print(" FRC HYBRID-PIC  --  derived plasma parameters")
        print("=" * 70)
        print(f" pressure balance : B_e = {B_e:8.4f} T")
        print(f" beta_null        = {beta:8.3f}   (=1 by construction)")
        print(f" w_ci = {w_ci:.3e} rad/s  (T_ci={2 * np.pi / w_ci:.2e} s)")
        print(f" dx   = {dx:.3e} m   dt = {dt:.3e} s")
        print("=" * 70)
    return out


# ============================================================================
# EQUILIBRIUM PROFILES  (2D Analytic Flux)
# ============================================================================
def _R_expr() -> str:
    return "sqrt(x*x + y*y)"


def g_expr(cfg: CFG, z0: float = 0.0) -> str:
    """2D racetrack separation function (g=0 at separatrix)."""
    R, R0, Zs, d = _R_expr(), cfg.R_NULL, cfg.ELONGATION * cfg.R_SEP, cfg.FIELD_SHARP
    return (f"( ({R})*({R})/({R0}*{R0}) "
            f"+ ((z-({z0}))/({Zs}))*((z-({z0}))/({Zs}))"
            f"*((z-({z0}))/({Zs}))*((z-({z0}))/({Zs})) - 1.0 ) / {d}")


def bz_profile_expr(cfg: CFG, z0: float = 0.0) -> str:
    return f"{cfg.B_EXT} * tanh({g_expr(cfg, z0)})"


def bz_total_expr(cfg: CFG) -> str:
    """Superimposed fields for merging without doubling asymptotic Be."""
    if cfg.SCENARIO == "merging":
        g_top = g_expr(cfg, cfg.MERGE_OFFSET)
        g_bot = g_expr(cfg, -cfg.MERGE_OFFSET)
        return f"{cfg.B_EXT} * (tanh({g_top}) + tanh({g_bot}) - 1.0)"
    return bz_profile_expr(cfg, 0.0)


def density_profile_expr(cfg: CFG, z0: float = 0.0) -> str:
    Bz_local = bz_profile_expr(cfg, z0)
    n0 = cfg.N_NULL
    return f"{n0} * max( {cfg.N_BG_FRAC}, 1.0 - ( ({Bz_local})/{cfg.B_EXT} )*( ({Bz_local})/{cfg.B_EXT} ) )"


def diamagnetic_uphi_expr(cfg: CFG, z0: float = 0.0) -> str:
    R = _R_expr();
    R0 = cfg.R_NULL;
    Be = cfg.B_EXT;
    d = cfg.FIELD_SHARP;
    n0 = cfg.N_NULL
    g = g_expr(cfg, z0)
    sech2 = f"(1.0/cosh({g}))*(1.0/cosh({g}))"
    dBz_dr = f"{Be} * {sech2} * 2.0*({R})/({R0}*{R0}*{d})"
    nfloor = f"max( {cfg.N_BG_FRAC * n0}, {density_profile_expr(cfg, z0)} )"
    return f"-( {dBz_dr} ) / ( {MU0} * ({nfloor}) * {Q_E} )"


# ============================================================================
# WARPX BUILD
# ============================================================================
def build_simulation(cfg: CFG, derived: dict):
    from pywarpx import picmi

    dx = derived["dx"]
    L_r = 1.3 * cfg.R_SEP
    L_z = 2.0 * cfg.ELONGATION * cfg.R_SEP
    BF = 8

    nr_rz = max(BF, int(np.ceil((L_r / dx) / BF)) * BF)
    nz = max(BF, int(np.ceil((L_z / dx) / BF)) * BF)

    if cfg.GEOMETRY == "RZ":
        n_modes = 1 if (cfg.DO_COULOMB or cfg.DO_FUSION) else 2
        grid = picmi.CylindricalGrid(
            number_of_cells=[nr_rz, nz],
            lower_bound=[0.0, -L_z / 2], upper_bound=[L_r, +L_z / 2],
            lower_boundary_conditions=["none", "dirichlet"],
            upper_boundary_conditions=["dirichlet", "dirichlet"],
            lower_boundary_conditions_particles=["none", "absorbing"],
            upper_boundary_conditions_particles=["absorbing", "absorbing"],
            n_azimuthal_modes=n_modes,
        )
    elif cfg.GEOMETRY == "3D":
        nr_3d = max(BF, int(np.ceil((2.0 * L_r / dx) / BF)) * BF)
        grid = picmi.Cartesian3DGrid(
            number_of_cells=[nr_3d, nr_3d, nz],
            lower_bound=[-L_r, -L_r, -L_z / 2], upper_bound=[L_r, L_r, +L_z / 2],
            lower_boundary_conditions=["dirichlet"] * 3,
            upper_boundary_conditions=["dirichlet"] * 3,
            lower_boundary_conditions_particles=["absorbing"] * 3,
            upper_boundary_conditions_particles=["absorbing"] * 3,
        )
    else:
        raise ValueError("GEOMETRY must be 'RZ' or '3D'")

    solver = picmi.HybridPICSolver(
        grid=grid, gamma=cfg.E_GAMMA, Te=cfg.T_E, n0=cfg.N_NULL,
        n_floor=cfg.N_BG_FRAC * cfg.N_NULL, plasma_resistivity=cfg.ETA,
        plasma_hyper_resistivity=cfg.ETA_HYPER, substeps=cfg.SUBSTEPS,
    )

    sim = picmi.Simulation(solver=solver, time_step_size=derived["dt"],
                           max_steps=cfg.MAX_STEPS, particle_shape="cubic", verbose=1)

    layout = picmi.PseudoRandomLayout(n_macroparticles_per_cell=cfg.PPC, grid=grid)
    R = _R_expr()
    species = []

    if cfg.SCENARIO == "merging":
        z0 = cfg.MERGE_OFFSET;
        vdr = cfg.MERGE_MACH * derived["v_A"]
        for sgn, tag in [(+1, "D_top"), (-1, "D_bot")]:
            n_expr = density_profile_expr(cfg, z0=sgn * z0)
            uphi = diamagnetic_uphi_expr(cfg, z0=sgn * z0)
            ux_expr = f"-({uphi})*( y/({R}+1e-30) )"
            uy_expr = f" ({uphi})*( x/({R}+1e-30) )"

            dist = picmi.AnalyticDistribution(
                density_expression=n_expr,
                momentum_expressions=[ux_expr, uy_expr, f"{-sgn * vdr}"],
                rms_velocity=[derived["v_thi"]] * 3,
                lower_bound=[None, None, 0.0 if sgn > 0 else None],
                upper_bound=[None, None, None if sgn > 0 else 0.0],
            )
            # FIX-5: Numeric charge/mass exclusively
            sp = picmi.Species(name=tag, charge=Q_E, mass=M_D, initial_distribution=dist)
            sim.add_species(sp, layout=layout);
            species.append(sp)
    else:
        n_expr = density_profile_expr(cfg, z0=0.0)
        uphi = diamagnetic_uphi_expr(cfg, z0=0.0)
        ux_expr = f"-({uphi})*( y/({R}+1e-30) )"
        uy_expr = f" ({uphi})*( x/({R}+1e-30) )"
        dist = picmi.AnalyticDistribution(
            density_expression=n_expr,
            momentum_expressions=[ux_expr, uy_expr, "0.0"],
            rms_velocity=[derived["v_thi"]] * 3,
        )
        ions = picmi.Species(name="deuterium_ions", charge=Q_E, mass=M_D, initial_distribution=dist)
        sim.add_species(ions, layout=layout);
        species.append(ions)

    Bz = bz_total_expr(cfg)
    if cfg.GEOMETRY == "3D" and cfg.SCENARIO == "equilibrium":
        eps = cfg.TILT_PERT;
        Zs = cfg.ELONGATION * cfg.R_SEP
        Bz = Bz.replace("y*y", f"(y - {eps}*{Zs}*sin(z*{np.pi}/{Zs}))**2")

    sim.add_applied_field(picmi.AnalyticInitialField(Bx_expression="0.0", By_expression="0.0", Bz_expression=Bz))
    fusion_products = _add_collisions(sim, species, cfg, layout)
    return sim, species, fusion_products, (nr_rz if cfg.GEOMETRY == "RZ" else nr_3d, nz)


def _add_collisions(sim, species, cfg: CFG, layout):
    from pywarpx import picmi
    fusion_products = {}

    if cfg.DO_COULOMB:
        try:
            cc = picmi.CoulombCollisions(name="dd_coulomb", species=[species[0], species[0]], ndt_supercycle=2)
            if getattr(sim, "collisions", None) is None:
                sim.collisions = []
            sim.collisions.append(cc)
        except Exception as e:
            print(f"[collisions] Coulomb setup skipped ({e})")

    if cfg.DO_FUSION:
        try:
            he3 = picmi.Species(name="helium3", charge=2.0 * Q_E, mass=M_HE3)
            neut = picmi.Species(name="neutron", charge=0.0, mass=M_N)
            trit = picmi.Species(name="tritium", charge=Q_E, mass=M_T)
            prot = picmi.Species(name="proton", charge=Q_E, mass=M_P)
            for s in (he3, neut, trit, prot):
                sim.add_species(s, layout=None)
            fusion_products = dict(he3=he3, neutron=neut, tritium=trit, proton=prot)
        except Exception as e:
            print(f"[collisions] Fusion products skipped ({e})")

    return fusion_products


def _set_species_types():
    from pywarpx import Particles
    SPECIES_TYPE = {
        "deuterium_ions": "deuterium", "D_top": "deuterium", "D_bot": "deuterium",
        "helium3": "helium3", "neutron": "neutron", "tritium": "tritium", "proton": "proton",
    }
    tagged = []
    for bucket in Particles.particles_list:
        nm = getattr(bucket, "instancename", None)
        st = SPECIES_TYPE.get(nm)
        if st is not None:
            bucket.species_type = st
            tagged.append(f"{nm}->{st}")
    print(f"[species] types set: {', '.join(tagged)}")


def register_dd_fusion(sim, cfg: CFG, ion_names: list[str], tag: str):
    from pywarpx import Collisions as WXC
    branches = [
        (f"dd_he3n_{tag}", "DeuteriumDeuteriumToNeutronHeliumFusion", ["helium3", "neutron"]),
        (f"dd_tp_{tag}", "DeuteriumDeuteriumToProtonTritiumFusion", ["tritium", "proton"]),
    ]
    existing = getattr(WXC.collisions, "collision_names", None)
    names = list(existing) if existing else []

    for name, ftype, products in branches:
        blk = WXC.newcollision(name)
        blk.type = "nuclearfusion"
        blk.fusion_type = ftype
        # FIX-9e: properly parenthesized
        blk.species = (ion_names + ion_names) if len(ion_names) == 1 else ion_names
        blk.product_species = products
        blk.event_multiplier = cfg.FUSION_MULT
        names.append(name)

    WXC.collisions.collision_names = names


# ============================================================================
# DIAGNOSTICS
# ============================================================================
class Diagnostics:
    # Pass 'sim' in so we can use the modern field API
    def __init__(self, sim, cfg: CFG, derived: dict, grid_shape, ion_names=None):
        self.sim = sim
        self.cfg = cfg; self.d = derived; self.nr, self.nz = grid_shape
        self.ion_names = ion_names or ["deuterium_ions"]
        self.t = []; self.E_B = []; self.E_K = []; self.Ti = []; self.flux_rec = []
        self.probe_Er = []
        self.step = 0
        self.frame_steps   = []
        self.frames_Bz     = []
        self.frames_Er     = []
        self._frame_stride = 1
        self._diag_count   = 0
        self.__name__ = "frc_diagnostics"

    def _get_first(self, names):
        """Modernized to use the new sim.fields API to avoid deprecation warnings."""
        for nm in names:
            try:
                # Use the new recommended PICMI method
                arr = self.sim.fields.get(nm)
                if arr is not None:
                    return arr[...]
            except Exception:
                pass
        raise AttributeError(f"None of {names} wrappers found on grid.")
    @staticmethod
    def _as_2d(arr):
        a = np.squeeze(np.asarray(arr, dtype=float))
        while a.ndim > 2:
            a = a[:, a.shape[1] // 2] if a.ndim == 3 else a[..., 0]
        return a

    def poloidal_flux(self, Bz, r):
        """psi(r,z) = integral_0^r B_z(r',z) 2pi r' dr' (left Riemann)"""
        integrand = Bz * (2 * np.pi * r)[:, None]
        return np.cumsum(integrand, axis=0) * (r[1] - r[0])

    def _gather_ion_velocities(self, include_weights=False):
        from pywarpx import particle_containers
        ux_all, uy_all, uz_all, w_all = [], [], [], []
        for nm in self.ion_names:
            try:
                pc = particle_containers.ParticleContainerWrapper(nm)
                ux_all += list(pc.get_particle_ux())
                uy_all += list(pc.get_particle_uy())
                uz_all += list(pc.get_particle_uz())
                if include_weights:
                    w_all += list(pc.get_particle_weight())
            except Exception:
                continue
        if not ux_all:
            raise RuntimeError("no ions found")
        if include_weights:
            return np.concatenate(ux_all), np.concatenate(uy_all), np.concatenate(uz_all), np.concatenate(w_all)
        return np.concatenate(ux_all), np.concatenate(uy_all), np.concatenate(uz_all)

    def _store_frames(self, Bz_raw, Er_raw):
        self._diag_count += 1
        if self._diag_count % self._frame_stride:
            return
        ds = max(1, int(self.cfg.FIELD_DOWNSAMPLE))
        self.frames_Bz.append(self._as_2d(Bz_raw)[::ds, ::ds].copy())
        self.frames_Er.append(None if Er_raw is None else self._as_2d(Er_raw)[::ds, ::ds].copy())
        self.frame_steps.append(self.step)
        if len(self.frames_Bz) > self.cfg.VIDEO_MAX_FRAMES:
            self.frames_Bz = self.frames_Bz[::2]
            self.frames_Er = self.frames_Er[::2]
            self.frame_steps = self.frame_steps[::2]
            self._frame_stride *= 2

    def __call__(self):
        self.step += 1

        # FIX-2: High-frequency probe sampling
        if self.step % self.cfg.PROBE_EVERY == 0:
            try:
                Er = self._get_first(("Er", "Ex"))
                Er2d = self._as_2d(Er)
                # FIX-9c: Full radial line-out
                self.probe_Er.append(Er2d[:, Er2d.shape[1] // 2].copy())
            except Exception:
                pass

        if self.step % self.cfg.DIAG_EVERY == 0:
            try:
                Bz = self._get_first(("Bz",))
                Bz2d = self._as_2d(Bz)

                # FIX-7: Volume-weighted field energy
                dr = (1.3 * self.cfg.R_SEP) / Bz2d.shape[0]
                dz = (2.0 * self.cfg.ELONGATION * self.cfg.R_SEP) / Bz2d.shape[1]
                if self.cfg.GEOMETRY == "RZ":
                    r = np.linspace(0.5 * dr, 1.3 * self.cfg.R_SEP - 0.5 * dr, Bz2d.shape[0])
                    dV = 2.0 * np.pi * r[:, None] * dr * dz
                else:
                    dV = self.d["dx"] ** 3

                self.E_B.append(float(np.sum((Bz2d ** 2) / (2 * MU0) * dV)))

                # FIX-7 & FIX-8: Weight-integrated EK and uz-isolated Ti
                ux, uy, uz, w = self._gather_ion_velocities(include_weights=True)
                self.E_K.append(0.5 * M_D * float(np.sum(w * (ux ** 2 + uy ** 2 + uz ** 2))))
                self.Ti.append(M_D * np.average((uz - np.average(uz, weights=w)) ** 2, weights=w) / Q_E)
            except Exception:
                self.E_B.append(np.nan);
                self.E_K.append(np.nan);
                self.Ti.append(np.nan)

            if self.cfg.SCENARIO == "merging":
                try:
                    r = np.linspace(0, 1.3 * self.cfg.R_SEP, Bz2d.shape[0])
                    psi = self.poloidal_flux(Bz2d, r)
                    self.flux_rec.append(float(psi[:, psi.shape[1] // 2].max()))
                except Exception:
                    pass

            if self.cfg.VIDEO:
                try:
                    Er_frame = self._get_first(("Er", "Ex"))
                    self._store_frames(Bz, Er_frame)
                except Exception:
                    try:
                        self._store_frames(Bz, None)
                    except Exception:
                        pass

            self.t.append(self.step)
            print(f"[step {self.step:6d}] E_B={self.E_B[-1]:.3e} Ti={self.Ti[-1]:.1f} eV")


# ============================================================================
# POST-PROCESSING
# ============================================================================
def post_process(diag: Diagnostics, derived: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation

    cfg = diag.cfg
    dt = derived["dt"]
    t = np.array(diag.t, dtype=float) * dt
    made = []

    def _writer_ext():
        try:
            if animation.writers.is_available("ffmpeg"):
                return animation.FFMpegWriter(fps=cfg.VIDEO_FPS, bitrate=2400), ".mp4"
        except Exception:
            pass
        return animation.PillowWriter(fps=cfg.VIDEO_FPS), ".gif"

    def _save_anim(fig, update, nframes, stem):
        if not cfg.VIDEO or nframes < 2:
            plt.close(fig);
            return
        writer, ext = _writer_ext()
        try:
            ani = animation.FuncAnimation(fig, update, frames=nframes, blit=False)
            ani.save(stem + ext, writer=writer, dpi=cfg.VIDEO_DPI)
            made.append(stem + ext)
        except Exception as e:
            print(f"[video] {stem} failed ({e})")
        finally:
            plt.close(fig)

    def _subsample(n_total, n_max):
        if n_total <= n_max: return np.arange(n_total)
        return np.unique(np.linspace(0, n_total - 1, n_max).astype(int))

    def _series_video(time, series, ylabel, title, stem, log=False):
        series = {k: np.asarray(v, float) for k, v in series.items() if
                  len(v) and np.isfinite(np.asarray(v, float)).any()}
        if not series: return
        nmax = max(len(v) for v in series.values())
        time = np.asarray(time, float)[:nmax]
        if len(time) < 2: return
        allv = np.concatenate([v[np.isfinite(v)] for v in series.values()])
        fig, ax = plt.subplots(figsize=(7, 4.2))
        lines = {k: (ax.semilogy if log else ax.plot)([], [], lw=1.6, label=k)[0] for k in series}
        ax.set_xlim(time[0], time[-1])
        if log:
            lo = max(float(allv[allv > 0].min()) if (allv > 0).any() else 1e-30, 1e-30)
            ax.set_ylim(lo * 0.5, float(allv.max()) * 2.0 + 1e-30)
        else:
            lo, hi = float(allv.min()), float(allv.max())
            pad = 0.05 * (hi - lo) or max(abs(hi), 1.0) * 0.05
            ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel("time [s]");
        ax.set_ylabel(ylabel)
        if len(series) > 1: ax.legend(loc="best")
        ttl = ax.set_title(title)
        idxs = _subsample(nmax, cfg.VIDEO_MAX_FRAMES)
        T_ci = derived["T_ci"]

        def update(i):
            n = idxs[i] + 1
            for k, v in series.items():
                m = min(n, len(v))
                lines[k].set_data(time[:m], v[:m])
            ttl.set_text(f"{title}   (t = {time[n - 1]:.3e} s = {time[n - 1] / T_ci:.2f} gyro-periods)")
            return tuple(lines.values())

        _save_anim(fig, update, len(idxs), stem)

    def _remove_contours(cs):
        if not cs: return
        try:
            cs.remove()
        except Exception:
            try:
                for c in cs.collections: c.remove()
            except Exception:
                pass

    def _field_video(frames, steps, label, title, stem, cmap, symmetric, overlay_psi=False):
        # FIX-9a: Synchronized step filtering
        valid = [(f, s) for f, s in zip(frames, steps) if f is not None]
        if len(valid) < 2: return
        frames_val = [v[0] for v in valid]
        steps_val = [v[1] for v in valid]

        nr_f, nz_f = frames_val[0].shape
        r = np.linspace(0.0, 1.3 * cfg.R_SEP, nr_f)
        z = np.linspace(-cfg.ELONGATION * cfg.R_SEP, +cfg.ELONGATION * cfg.R_SEP, nz_f)
        vabs = max(float(np.nanmax(np.abs(f))) for f in frames_val) or 1.0
        vmin, vmax = (-vabs, vabs) if symmetric else (min(float(np.nanmin(f)) for f in frames_val), vabs)
        fig, ax = plt.subplots(figsize=(8.5, 4.0))
        mesh = ax.pcolormesh(z, r, frames_val[0], cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
        fig.colorbar(mesh, ax=ax, label=label)
        ax.set_xlabel("z [m]");
        ax.set_ylabel("r [m]")
        ttl = ax.set_title(title)
        state = {"cs": None}
        idxs = _subsample(len(frames_val), cfg.VIDEO_MAX_FRAMES)
        T_ci = derived["T_ci"]

        def update(i):
            j = idxs[i]
            F = frames_val[j]
            mesh.set_array(F.ravel())
            if overlay_psi:
                _remove_contours(state["cs"])
                try:
                    state["cs"] = ax.contour(z, r, diag.poloidal_flux(F, r), levels=12, colors="k", linewidths=0.5)
                except Exception:
                    state["cs"] = None
            tt = steps_val[j] * dt
            ttl.set_text(f"{title}   t = {tt:.3e} s ({tt / T_ci:.2f} gyro-periods)")
            return (mesh,)

        _save_anim(fig, update, len(idxs), stem)

    # 1. energy
    plt.figure()
    if len(diag.E_B): plt.semilogy(t[:len(diag.E_B)], np.maximum(diag.E_B, 1e-30), label="magnetic")
    if any(np.isfinite(diag.E_K)): plt.semilogy(t[:len(diag.E_K)], np.maximum(diag.E_K, 1e-30), label="ion kinetic")
    plt.xlabel("time [s]");
    plt.ylabel("energy [J]");
    plt.legend()
    plt.title("Energy partition")
    plt.savefig("frc_energy.png", dpi=140);
    plt.close();
    made.append("frc_energy.png")
    _series_video(t, {"magnetic": np.maximum(np.array(diag.E_B, float), 1e-30),
                      "ion kinetic": np.maximum(np.array(diag.E_K, float), 1e-30)},
                  "energy [J]", "Energy partition", "frc_energy", log=True)

    # 2. Ti
    if any(np.isfinite(diag.Ti)):
        plt.figure();
        plt.plot(t[:len(diag.Ti)], diag.Ti)
        plt.xlabel("time [s]");
        plt.ylabel(r"$T_i$ [eV]")
        plt.title("Ion temperature (var(u_z) decoupled from u_phi shear)")
        plt.savefig("frc_ion_heating.png", dpi=140);
        plt.close();
        made.append("frc_ion_heating.png")
        _series_video(t, {r"$T_i$": diag.Ti}, r"$T_i$ [eV]", "Ion temperature", "frc_ion_heating", log=False)

    # 3. Bernstein
    if len(diag.probe_Er) > 16:
        probe_arr = np.array(diag.probe_Er, float)
        sig = probe_arr[:, probe_arr.shape[1] // 2]
        sig -= sig.mean()
        dt_probe = dt * cfg.PROBE_EVERY
        f_ci = derived["w_ci"] / (2 * np.pi)
        spec_full = np.abs(np.fft.rfft(sig)) ** 2
        freq_full = np.fft.rfftfreq(len(sig), d=dt_probe)

        plt.figure();
        plt.semilogy(freq_full / f_ci, spec_full)
        for nharm in range(1, 6): plt.axvline(nharm, ls="--", lw=0.7)
        plt.xlabel(r"$\omega/\omega_{ci}$");
        plt.ylabel(r"$|E_r(\omega)|^2$")
        plt.title("Ion Bernstein harmonics")
        plt.xlim(0, 6);
        plt.savefig("frc_bernstein.png", dpi=140);
        plt.close();
        made.append("frc_bernstein.png")

        if cfg.VIDEO and len(sig) > 32:
            n0 = 32
            ends = np.unique(np.linspace(n0, len(sig), min(cfg.VIDEO_MAX_FRAMES, len(sig) - n0 + 1)).astype(int))
            smax = float(spec_full.max()) or 1.0
            fig, ax = plt.subplots(figsize=(7, 4.2))
            (line,) = ax.semilogy([], [], lw=1.2)
            for nharm in range(1, 6): ax.axvline(nharm, ls="--", lw=0.7, color="gray")
            ax.set_xlim(0, 6);
            ax.set_ylim(max(smax * 1e-8, 1e-30), smax * 2.0)
            ax.set_xlabel(r"$\omega/\omega_{ci}$");
            ax.set_ylabel(r"$|E_r(\omega)|^2$")
            ttl = ax.set_title("")

            def upd_bern(i):
                n = ends[i]
                line.set_data(np.fft.rfftfreq(n, d=dt_probe) / f_ci, np.abs(np.fft.rfft(sig[:n])) ** 2)
                ttl.set_text(f"Ion Bernstein spectrum (t < {n * dt_probe:.3e} s)")
                return (line,)

            _save_anim(fig, upd_bern, len(ends), "frc_bernstein")

    # 4. Flux
    if len(diag.flux_rec):
        plt.figure();
        plt.plot(t[:len(diag.flux_rec)], diag.flux_rec)
        plt.xlabel("time [s]");
        plt.ylabel(r"$\psi$ at midplane [Wb]")
        plt.savefig("frc_reconnection.png", dpi=140);
        plt.close();
        made.append("frc_reconnection.png")
        _series_video(t, {r"$\psi$ midplane": diag.flux_rec}, r"$\psi$ at midplane [Wb]", "Poloidal flux at X-point",
                      "frc_reconnection", log=False)

    # 5. Fields
    if cfg.VIDEO and len(diag.frames_Bz) >= 2:
        _field_video(diag.frames_Bz, diag.frame_steps, r"$B_z$ [T]", r"$B_z$ + poloidal-flux contours", "frc_fields",
                     cmap="RdBu_r", symmetric=True, overlay_psi=(cfg.GEOMETRY == "RZ"))
    if cfg.VIDEO and any(f is not None for f in diag.frames_Er):
        _field_video(diag.frames_Er, diag.frame_steps, r"$E_r$ [V/m]", r"$E_r(r,z)$", "frc_er_fields", cmap="PuOr_r",
                     symmetric=True, overlay_psi=False)

    print("Post-processing complete. Wrote:")
    for f in made: print(f"  {f}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()
    if args.no_video: CFG.VIDEO = False

    derived = derive_parameters(CFG, verbose=True)
    if args.report: return

    from pywarpx import callbacks
    sim, species, fusion_products, grid_shape = build_simulation(CFG, derived)
    ion_names = [sp.name for sp in species]

    # Pass the 'sim' object as the first argument here:
    diag = Diagnostics(sim, CFG, derived, grid_shape, ion_names=ion_names)
    callbacks.installafterstep(diag)

    if CFG.DO_FUSION and fusion_products:
        try:
            sim.initialize_inputs()
            _set_species_types()
            if CFG.SCENARIO == "merging":
                # FIX-3: Parameterized cross-lobe fusion registration
                register_dd_fusion(sim, CFG, [ion_names[0], ion_names[0]], "tt")
                if len(ion_names) > 1:
                    register_dd_fusion(sim, CFG, [ion_names[1], ion_names[1]], "bb")
                    register_dd_fusion(sim, CFG, [ion_names[0], ion_names[1]], "tb")
            else:
                register_dd_fusion(sim, CFG, [ion_names[0], ion_names[0]], "main")
        except Exception as e:
            print(f"[fusion] ParmParse registration failed ({e})")

    sim.step(CFG.MAX_STEPS)
    post_process(diag, derived)


if __name__ == "__main__":
    main()

