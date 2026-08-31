"""
slip_gate_diagnostic.py

NOT a file to run. This is the diagnostic method to PASTE into
physics_losses.py (inside class PhysicsLosses, right after h_dissipation),
plus the 4-line wiring snippet for train_force_gns.py at the bottom.

WHAT QUESTION IT ANSWERS
-----------------------
h_dissipation multiplies its residual by

    slip_gate = sigmoid((|v_t| - slip_v0) / slip_tau)

with slip_v0 = 1e-3 m/STEP. At dt = 1/148 s that is 0.148 m/s - a fairly
high bar for a cube that is sliding to a stop. If most contact frames sit
below it, the gate is ~0, h_dissipation contributes nothing whatever weight
you give it, and mu (which gets gradient ONLY through this term) is being
fit from a small, biased subset of frames.

That would explain three things at once, all of which are in your results:
  - the F_w_diss_0 ablation showed removing w_diss costs nothing
  - recovered_mu sits at 0.152-0.158 against a true 0.198
  - recovered_mu agrees to +/-0.0002 across independent seeds, which is far
    too rigid to be a real data fit

This prints the gate occupancy and, crucially, COUNTERFACTUAL occupancies at
lower slip_v0 - so one epoch tells you whether lowering slip_v0 fixes it or
whether the term is unsalvageable for a different reason.
"""

# ======================================================================
# PASTE THIS INTO physics_losses.py, inside class PhysicsLosses,
# immediately after h_dissipation().
# ======================================================================

DIAGNOSTIC_METHOD = r'''
    # ------------------------------------------------------------------
    # DIAGNOSTIC - not a loss term, never called in the training graph.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def slip_gate_report(self, phi_contact, c_w, v_node, wall_n, dt=None):
        """Is h_dissipation awake? Mirrors h_dissipation's gating exactly.

        h_dissipation weights every node by (c_w * slip_gate). If the
        slip_gate factor is ~0 across the batch, the term contributes
        nothing at ANY weight and mu - whose only gradient path is this
        term - is being fit from whatever sliver of frames does get through.

        Returns a dict; see fmt_slip_gate_report() for the one-line print.
        phi_contact/c_w/v_node/wall_n: the SAME tensors you pass to
        compute_step_terms. dt (s) is optional and only converts the
        m/step figures to m/s for readability.
        """
        v = v_node.detach()
        v_t = v - (v * wall_n).sum(-1, keepdim=True) * wall_n
        speed = v_t.norm(dim=-1, keepdim=True)                 # (B,N,1) m/step
        gate = torch.sigmoid((speed - self.slip_v0) / self.slip_tau)

        w_c = c_w.detach()
        contact_mass = w_c.sum().clamp_min(self.eps)
        # This ratio IS the multiplier on h_dissipation's effective size.
        gate_frac = float((w_c * gate).sum() / contact_mass)

        # Slip-speed distribution over nodes that are actually in contact,
        # which is the population the gate is deciding about.
        in_contact = (w_c > 0.5).squeeze(-1)
        s = speed.squeeze(-1)[in_contact]
        if s.numel() == 0:
            pct = {q: float('nan') for q in (10, 50, 90, 99)}
        else:
            qs = torch.tensor([0.10, 0.50, 0.90, 0.99], device=s.device,
                              dtype=s.dtype)
            vals = torch.quantile(s, qs)
            pct = {q: float(x) for q, x in zip((10, 50, 90, 99), vals)}

        # Counterfactuals: what would gate_frac be at a lower threshold?
        # This is the number that decides whether slip_v0 is the real knob.
        cf = {}
        for div in (3.0, 10.0, 30.0):
            g2 = torch.sigmoid((speed - self.slip_v0 / div) / self.slip_tau)
            cf[div] = float((w_c * g2).sum() / contact_mass)

        # mu implied by the model's OWN predicted forces on sliding nodes.
        # Coulomb says ||phi_t|| = mu * phi_n while sliding, so this is the
        # mu the force decomposition is currently consistent with -
        # independent of the learnable mu parameter.
        phi_n = (phi_contact.detach() * wall_n).sum(-1, keepdim=True)
        phi_t = phi_contact.detach() - phi_n * wall_n
        wg = w_c * gate
        num = (wg * phi_t.norm(dim=-1, keepdim=True)).sum()
        den = (wg * phi_n.clamp_min(0.0)).sum()
        mu_implied = float(num / den) if float(den) > self.eps else float('nan')

        return dict(gate_frac=gate_frac,
                    slip_v0=float(self.slip_v0),
                    slip_tau=float(self.slip_tau),
                    pct=pct, counterfactual=cf,
                    mu_implied=mu_implied, mu_param=float(self.mu),
                    n_contact_nodes=float(contact_mass), dt=dt)


def fmt_slip_gate_report(r):
    """Two lines for the epoch log. Import alongside PhysicsLosses."""
    dt = r["dt"]
    to_ms = (lambda x: x / dt) if dt else (lambda x: float('nan'))
    u = "m/s" if dt else "m/step"
    conv = to_ms if dt else (lambda x: x)
    p = r["pct"]
    cf = r["counterfactual"]
    return (
        f"  Slip gate | OPEN {r['gate_frac']:6.1%} of contact weight  "
        f"| v0={conv(r['slip_v0']):.3f} {u}  "
        f"| contact slip p10/p50/p90/p99 = "
        f"{conv(p[10]):.3f}/{conv(p[50]):.3f}/{conv(p[90]):.3f}/{conv(p[99]):.3f} {u}\n"
        f"            | if v0 were /3: {cf[3.0]:5.1%}   /10: {cf[10.0]:5.1%}   "
        f"/30: {cf[30.0]:5.1%}   "
        f"| mu implied by predicted forces = {r['mu_implied']:.3f} "
        f"(mu param = {r['mu_param']:.3f})"
    )
'''


# ======================================================================
# PASTE THIS INTO train_force_gns.py, next to the existing
# "Physics terms (raw) | ..." print, so it runs once per epoch on one batch.
# ======================================================================

WIRING_SNIPPET = r'''
# --- at the top, next to the PhysicsLosses import ---
from physics_losses import PhysicsLosses, fmt_slip_gate_report

# --- inside the unroll, on the FIRST batch of the epoch only ---
# (same tensors you already hand to phys.compute_step_terms)
if is_first_batch_of_epoch and step_idx == 0:
    slip_report = phys.slip_gate_report(phi_contact, c_w, v_node, wall_n, dt=dt)

# --- next to the existing "Physics terms (raw)" print ---
if slip_report is not None:
    print(fmt_slip_gate_report(slip_report))
'''


if __name__ == "__main__":
    print(__doc__)
    print("=" * 74)
    print("METHOD -> physics_losses.py (inside class PhysicsLosses)")
    print("=" * 74)
    print(DIAGNOSTIC_METHOD)
    print("=" * 74)
    print("WIRING -> train_force_gns.py")
    print("=" * 74)
    print(WIRING_SNIPPET)
