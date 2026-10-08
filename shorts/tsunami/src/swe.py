"""Shallow-water solver for the tsunami: depth h and discharges (hu, hv) on a uniform grid over a bed b (terrain plus
whatever buildings still stand), finite volumes with HLL fluxes and hydrostatic reconstruction (Audusse et al.), so
the sea at rest stays at rest over any bed, fronts run up dry land without going negative, and a bore stays sharp.
Manning friction, a wavemaker on the sea edge (a long wave of the round's height rolling in), open edges elsewhere.

Grid: arrays are indexed [j, i] = [y, x]; cell (j, i) is centred at (x0 + (i + 0.5) dx, y0 + (j + 0.5) dx). The sea
is at the high-y edge.

Foam is a scalar the flow carries along: made where the water piles up (converging, breaking fronts, against
walls), advected, and fading.
"""
import numpy as np

G = 9.81
DRY = 1e-3


class SWE:
    def __init__(self, bed, dx, x0, y0, sea_level=0.0, manning=0.022, h_init=None):
        self.b = np.asarray(bed, np.float32).copy()
        self.dx = float(dx)
        self.x0, self.y0 = float(x0), float(y0)
        self.ny, self.nx = self.b.shape
        self.sea = float(sea_level)
        if h_init is None:
            h_init = np.maximum(0.0, sea_level - self.b)
        self.h = np.asarray(h_init, np.float32).copy()
        self.hu = np.zeros_like(self.h)
        self.hv = np.zeros_like(self.h)
        self.foam = np.zeros_like(self.h)
        self.n = manning
        self.t = 0.0
        self.wave = None            # function t -> amplitude of the incoming wave at the sea edge
        self.umax = 60.0

    # -- helpers -------------------------------------------------------------------------------------------
    def velocities(self, h=None, hu=None, hv=None):
        h = self.h if h is None else h
        hu = self.hu if hu is None else hu
        hv = self.hv if hv is None else hv
        wet = h > DRY
        u = np.where(wet, hu / np.maximum(h, DRY), 0.0)
        v = np.where(wet, hv / np.maximum(h, DRY), 0.0)
        return u, v

    def eta(self):
        return self.h + self.b

    def _padded(self):
        """State with one ghost cell all round: open (copied) edges, the wavemaker on the sea (high-y) edge."""
        h = np.pad(self.h, 1, mode='edge')
        hu = np.pad(self.hu, 1, mode='edge')
        hv = np.pad(self.hv, 1, mode='edge')
        b = np.pad(self.b, 1, mode='edge')
        if self.wave is not None:
            a = float(self.wave(self.t))
            hs = np.maximum(self.sea - b[-1, :], 0.0)
            hw = np.maximum(hs + a, 0.0)
            # a long wave moving towards -y: u = -2 (sqrt(g h) - sqrt(g h0)) (Riemann invariant)
            vv = -2.0 * (np.sqrt(G * hw) - np.sqrt(G * hs))
            h[-1, :] = hw
            hu[-1, :] = 0.0
            hv[-1, :] = hw * vv
        return h, hu, hv, b

    @staticmethod
    def _hll(hL, huL, hvL, hR, huR, hvR, normal_u):
        """HLL flux across faces whose normal is x (normal_u=True: u is the normal velocity) or y."""
        wl, wr = hL > DRY, hR > DRY
        uL = np.where(wl, huL / np.maximum(hL, DRY), 0.0)
        vL = np.where(wl, hvL / np.maximum(hL, DRY), 0.0)
        uR = np.where(wr, huR / np.maximum(hR, DRY), 0.0)
        vR = np.where(wr, hvR / np.maximum(hR, DRY), 0.0)
        if normal_u:
            qL, tL, qR, tR = uL, vL, uR, vR
        else:
            qL, tL, qR, tR = vL, uL, vR, uR
        cL, cR = np.sqrt(G * hL), np.sqrt(G * hR)
        sL = np.minimum(qL - cL, qR - cR)
        sR = np.maximum(qL + cL, qR + cR)
        # fluxes: mass, normal momentum, tangential momentum
        FL = (hL * qL, hL * qL * qL + 0.5 * G * hL * hL, hL * qL * tL)
        FR = (hR * qR, hR * qR * qR + 0.5 * G * hR * hR, hR * qR * tR)
        UL = (hL, hL * qL, hL * tL)
        UR = (hR, hR * qR, hR * tR)
        den = sR - sL
        den = np.where(np.abs(den) < 1e-9, 1e-9, den)
        out = []
        for k in range(3):
            f = (sR * FL[k] - sL * FR[k] + sL * sR * (UR[k] - UL[k])) / den
            f = np.where(sL >= 0, FL[k], np.where(sR <= 0, FR[k], f))
            f = np.where(~wl & ~wr, 0.0, f)
            out.append(f)
        return out, max(float(np.max(np.abs(sL))), float(np.max(np.abs(sR))))

    # -- one step --------------------------------------------------------------------------------------------
    def max_dt(self, cfl=0.45):
        u, v = self.velocities()
        c = np.sqrt(G * self.h)
        s = max(float(np.max(np.abs(u) + c)), float(np.max(np.abs(v) + c)), 1e-6)
        if self.wave is not None:
            a = abs(float(self.wave(self.t)))
            s = max(s, 3 * np.sqrt(G * (a + max(0.0, self.sea - float(self.b[-1].min())) + 1e-6)))
        return cfl * self.dx / s

    def step(self, dt):
        h, hu, hv, b = self._padded()
        eta = h + b
        dx = self.dx
        # x faces: between columns i and i+1 of the padded arrays, rows 1..ny
        r = slice(1, -1)
        U, Vv = self.velocities(h, hu, hv)
        bL, bR = b[r, :-1], b[r, 1:]
        bs = np.maximum(bL, bR)
        hL = np.maximum(0.0, eta[r, :-1] - bs)
        hR = np.maximum(0.0, eta[r, 1:] - bs)
        uL, vL = U[r, :-1], Vv[r, :-1]
        uR, vR = U[r, 1:], Vv[r, 1:]
        (Fh, Fn, Ft), sx = self._hll(hL, hL * uL, hL * vL, hR, hR * uR, hR * vR, True)
        # well-balanced pressure corrections for the cell on each side of the face
        corrL = 0.5 * G * (h[r, :-1] ** 2 - hL ** 2)
        corrR = 0.5 * G * (h[r, 1:] ** 2 - hR ** 2)
        # y faces: between rows j and j+1, columns 1..nx
        c = slice(1, -1)
        bD, bU = b[:-1, c], b[1:, c]
        bs = np.maximum(bD, bU)
        hD = np.maximum(0.0, eta[:-1, c] - bs)
        hU = np.maximum(0.0, eta[1:, c] - bs)
        uD, vD = U[:-1, c], Vv[:-1, c]
        uU, vU = U[1:, c], Vv[1:, c]
        (Gh, Gn, Gt), sy = self._hll(hD, hD * uD, hD * vD, hU, hU * uU, hU * vU, False)
        corrD = 0.5 * G * (h[:-1, c] ** 2 - hD ** 2)
        corrU = 0.5 * G * (h[1:, c] ** 2 - hU ** 2)
        k = dt / dx
        # cell i gets the face on its right (i+1/2: index i+1 in face arrays, as the left cell) and on its left
        # (i-1/2: index i, as the right cell)
        dh = -(Fh[:, 1:] - Fh[:, :-1]) - (Gh[1:, :] - Gh[:-1, :])
        dhu = -((Fn[:, 1:] + corrL[:, 1:]) - (Fn[:, :-1] + corrR[:, :-1])) - (Gt[1:, :] - Gt[:-1, :])
        dhv = -(Ft[:, 1:] - Ft[:, :-1]) - ((Gn[1:, :] + corrD[1:, :]) - (Gn[:-1, :] + corrU[:-1, :]))
        hn = self.h + k * dh
        hun = self.hu + k * dhu
        hvn = self.hv + k * dhv
        hn = np.maximum(hn, 0.0)
        dry = hn < DRY
        hun[dry] = 0.0
        hvn[dry] = 0.0
        # Manning friction, implicit
        u, v = self.velocities(hn, hun, hvn)
        sp = np.sqrt(u * u + v * v)
        fr = 1.0 + dt * G * self.n ** 2 * sp / np.maximum(hn, 0.05) ** (4.0 / 3.0)
        hun /= fr
        hvn /= fr
        # keep velocities sane at thin fronts
        u, v = self.velocities(hn, hun, hvn)
        sp = np.sqrt(u * u + v * v)
        cap = np.minimum(1.0, self.umax / np.maximum(sp, 1e-9))
        hun *= cap
        hvn *= cap
        # foam: made where the water rises fast (the breaking front), where it converges and where it's fast
        # against obstacles; carried along and fading
        rise = ((hn + self.b) - (self.h + self.b)) / max(dt, 1e-6)
        self._foam(dt, hn, u, v, rise)
        self.h, self.hu, self.hv = hn, hun, hvn
        self.t += dt

    def _foam(self, dt, h, u, v, rise):
        dx = self.dx
        du = np.zeros_like(u)
        dv = np.zeros_like(v)
        du[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2 * dx)
        dv[1:-1, :] = (v[2:, :] - v[:-2, :]) / (2 * dx)
        div = du + dv
        sp = np.sqrt(u * u + v * v)
        wet = h > 0.05
        scale = getattr(self, 'foam_scale', 1.0)          # rise rates scale with the size of the wave
        gen = np.clip(rise / scale - 1.0, 0.0, 6.0) * 0.7
        gen += np.clip(-div / scale ** 0.5 - 0.25, 0.0, 3.0) * 0.35 * np.clip(sp / (3.0 * scale ** 0.5), 0.0, 1.5)
        if not hasattr(self, '_gb'):
            gb = np.zeros_like(h)
            gb[1:-1, 1:-1] = np.hypot(self.b[1:-1, 2:] - self.b[1:-1, :-2],
                                      self.b[2:, 1:-1] - self.b[:-2, 1:-1]) / (2 * dx)
            self._gb = np.clip(gb, 0, 2)
        gen += self._gb * np.clip(sp / (4.0 * scale ** 0.5) - 0.5, 0, 2) * 0.35
        f = self.foam + dt * gen * wet
        # semi-Lagrangian advection
        if not hasattr(self, '_grid'):
            self._grid = np.mgrid[0:self.ny, 0:self.nx].astype(np.float32)
        jj, ii = self._grid
        si = np.clip(ii - u * dt / dx, 0, self.nx - 1.001)
        sj = np.clip(jj - v * dt / dx, 0, self.ny - 1.001)
        i0, j0 = si.astype(int), sj.astype(int)
        fi, fj = si - i0, sj - j0
        a = f[j0, i0] * (1 - fi) + f[j0, i0 + 1] * fi
        c = f[j0 + 1, i0] * (1 - fi) + f[j0 + 1, i0 + 1] * fi
        f = a * (1 - fj) + c * fj
        f *= np.exp(-dt / getattr(self, 'foam_life', 2.2))
        self.foam = np.clip(f, 0.0, 1.3) * wet

    def run_to(self, t_end, cfl=0.45, max_dt=0.05, callback=None):
        while self.t < t_end - 1e-9:
            dt = min(self.max_dt(cfl), max_dt, t_end - self.t)
            self.step(dt)
            if callback is not None:
                callback(self, dt)
