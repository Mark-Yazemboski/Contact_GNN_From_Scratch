"""
physics_losses.py

NEW FILE - the physics-informed violation losses from the proposal (Eq. 5-6),
mapped onto the force architecture. Each term is its own function so it can be
explained, ablated, and weighted independently.

======================================================================
MAPPING FROM THE PROPOSAL TO THIS IMPLEMENTATION
======================================================================

Proposal Eq. (5) has three terms. Where each one lives here:

  h_diss   "frictional forces must maximize power loss"
           -> h_dissipation() below. The proposal's expression
              || ||J_t v'|| lam_t + lam_n J_t v' ||  is zero exactly when the
              tangential impulse is anti-parallel to slip with magnitude tied
              to the normal impulse - i.e. kinetic Coulomb friction
              phi_t = -mu * phi_n * v_hat. We implement that zero-set
              directly, with mu made EXPLICIT and (by default) LEARNABLE, so
              the model recovers the friction coefficient as a byproduct -
              the same "recovered the physical parameter" story as the drag
              coefficient. Gated on slip speed: static friction may sit
              anywhere inside the cone, so the equality applies only while
              sliding.

  h_pen    "contact impulses must remain non-negative"
           -> ARCHITECTURAL. The normal force goes through a softplus in
              force_gns.py, so min(0, phi_n)^2 == 0 by construction. There is
              no loss term because violation is impossible, not merely
              discouraged. gamma_2 is not needed.

  h_smooth "regularize the predicted fluid forces ... to promote smooth
            fluid force distributions"
           -> Two terms, because our fluid head is a single COM wrench, not a
              per-node field (see force_gns.py for why):
              (a) h_fluid_anchor():   the fluid wrench should match the
                  ANALYTIC drag law k|u|u evaluated at the measured relative
                  wind (and the fluid torque should be ~0, as it is in
                  MuJoCo). This is the physics-infused version of
                  "regularize the fluid forces": shrink toward the law, not
                  toward zero.
              (b) h_fluid_temporal_smooth(): the fluid force must vary
                  SMOOTHLY IN TIME. Drag is a smooth function of relative
                  wind, which changes slowly; contact events are the jumpy
                  thing. With space collapsed to a point, "smooth
                  distribution" becomes smoothness along the trajectory.
                  Requires multistep >= 2 (needs consecutive predictions).

  plus     "sparsity regularization for concentrated forces" (Fig. 1)
           -> h_contact_sparsity(): L1 on contact force magnitudes.

Overall loss (Eq. 6):  L = L_pred + sum_j gamma_j h_j
The gammas are the w_* weights in run_force_multi_step.py.

======================================================================
WHY THESE TERMS, GIVEN WHAT WE MEASURED
======================================================================
The wrench-label evaluation showed the failure precisely: the fluid channel
carries a ~0.2 mg force during contact - the size of the friction force
mu*m*g - identically at every wind level, while free-flight drag is predicted
well. The loss only observes the net wrench (6 numbers) but the model outputs
30, so the split is underdetermined and the optimizer parks friction in the
easiest channel. These terms remove that degeneracy from both sides:
h_fluid_anchor pins what fluid IS ALLOWED to be (the drag law),
h_fluid_temporal_smooth pins how it may CHANGE (slowly), and h_dissipation
gives the displaced friction a correctly-structured home in the contact
channel (anti-parallel to slip, proportional to local normal force, one
global mu).

======================================================================
NORMALIZATION
======================================================================
Every force-like residual is divided by phi_g = g*dt^2, the specific weight
of the cube per step^2 - so a raw value of 1.0 means "a violation the size of
gravity". Torque-like residuals are divided by the empirical angular
acceleration std. This makes the printed raw magnitudes interpretable and the
weights transferable across datasets. Calibration rule: after one epoch, read
the printed raws and set each weight so (weight * raw) is 1-10% of the
position loss.

Slip velocities and gates are DETACHED: the physics terms constrain the
predicted forces given the observed motion; they must not create an incentive
to change the motion to relax the constraint.
"""

import torch
import torch.nn as nn


class PhysicsLosses(nn.Module):
    """Holds the physics-loss state (the friction coefficient mu) and exposes
    one method per violation term. Instantiate once in training, move to the
    device, and include .parameters() in the optimizer when mu is learnable.

    mu modes:
      fixed_mu = None, learn_mu = True   -> mu is a learnable parameter
                                            (init mu_init), recovered from
                                            data. Parameterized as log(mu) so
                                            it stays positive.
      fixed_mu = <float>                 -> mu clamped to the known value
                                            (stronger physics infusion; use
                                            for the ablation arm).
    """

    def __init__(self, phi_g, ang_scale_vec,
                 mu_init=0.2, learn_mu=True, fixed_mu=None,
                 slip_v0=1e-3, slip_tau=1e-4, eps=1e-9):
        super().__init__()
        self.register_buffer("phi_g", torch.as_tensor(float(phi_g)))
        self.register_buffer("ang_scale_vec",
                             torch.as_tensor(ang_scale_vec, dtype=torch.float32))
        self.fixed_mu = fixed_mu
        if fixed_mu is None:
            log_mu = torch.log(torch.tensor(float(mu_init)))
            if learn_mu:
                self.log_mu = nn.Parameter(log_mu)
            else:
                self.register_buffer("log_mu", log_mu)
        self.slip_v0 = slip_v0        # m/step: slip-speed gate center
        self.slip_tau = slip_tau      # m/step: gate softness
        self.eps = eps

    @property
    def mu(self):
        if self.fixed_mu is not None:
            return torch.as_tensor(self.fixed_mu, device=self.phi_g.device)
        return torch.exp(self.log_mu)

    # ------------------------------------------------------------------
    # h_diss  (proposal Eq. 5, first term)
    # ------------------------------------------------------------------
    def h_dissipation(self, phi_contact, c_w, v_node, wall_n):
        """Kinetic Coulomb friction on sliding contact nodes.

        Zero exactly when  phi_t = -mu * phi_n * v_hat_t  on every sliding
        contact node: friction opposes slip (=> can never add energy, the
        proposal's 'maximize power loss') AND its magnitude is mu times the
        LOCAL normal force. This couples the tangential channel to the normal
        channel through one global mu - structure the position loss alone
        cannot supply.

        Gating: contact weight c_w (geometric) x a soft slip gate
        sigma((|v_t| - v0)/tau). Static contact (settled cube) is NOT forced
        to the cone boundary - static friction may be anything inside it.

        phi_contact: (B, N, 3) contact specific forces (m/step^2)
        c_w:         (B, N, 1) contact weight
        v_node:      (B, N, 3) per-node velocity (m/step) - detached inside
        wall_n:      (3,) unit wall normal
        Returns a scalar: weighted mean squared residual in units of phi_g.
        """
        v = v_node.detach()
        v_t = v - (v * wall_n).sum(-1, keepdim=True) * wall_n
        speed = v_t.norm(dim=-1, keepdim=True)
        v_hat = v_t / (speed + self.eps)
        slip_gate = torch.sigmoid((speed - self.slip_v0) / self.slip_tau)

        phi_n = (phi_contact * wall_n).sum(-1, keepdim=True)      # signed normal
        phi_t = phi_contact - phi_n * wall_n

        resid = (phi_t + self.mu * phi_n * v_hat) / self.phi_g    # (B, N, 3)
        # Normalize by CONTACT weight only; the slip gate lives in the
        # numerator. If it were in the denominator too, a batch where every
        # node is equally (barely) gated would cancel the gate entirely and
        # static contact would be penalized at full strength. The gate is
        # sharp (tau default 1e-4 m/step) and DETACHED, so sharpness costs no
        # gradient pathology.
        w = (c_w * slip_gate).detach()                            # (B, N, 1)
        norm = c_w.detach().sum() + self.eps
        return (w * resid.pow(2).sum(-1, keepdim=True)).sum() / norm

    # ------------------------------------------------------------------
    # h_pen  (proposal Eq. 5, second term) - architectural, no code needed.
    # softplus in force_gns.assemble_contact_forces makes phi_n >= 0 always,
    # so min(0, phi_n)^2 == 0 by construction.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # h_smooth part (a): anchor the fluid wrench to the analytic drag law
    # ------------------------------------------------------------------
    def h_fluid_anchor(self, a_fluid_total, drag_target):
        """The fluid FORCE must be what aerodynamics permits: the TOTAL
        predicted fluid acceleration (learned residual plus baseline, if
        enabled) is pulled toward the analytic quadratic law k|u|u evaluated
        at the MEASURED relative wind. This is shrinkage toward the physics,
        not toward zero - with the drag baseline on it reduces to keeping the
        residual small (PIROM: the physics carries, the network corrects).

        a_fluid_total: (B, 3) m/step^2 (learned + baseline)
        drag_target:   (B, 3) m/step^2, analytic law at measured u - DETACHED
                       by the caller (it is a target, not a pathway).
        """
        return ((a_fluid_total - drag_target) / self.phi_g).pow(2).sum(-1).mean()

    def h_fluid_torque(self, alpha_fluid):
        """The fluid TORQUE must be ~zero: MuJoCo applies no meaningful fluid
        torque on this body, so every real angular change must come from
        contact lever arms. Separate from the force anchor (own weight, own
        printed raw) because the two can be in very different regimes - a
        rotation event the contact channel has not yet learned to explain
        shows up HERE, and you want to see that, not have it silently averaged
        into the force number."""
        return (alpha_fluid / self.ang_scale_vec).pow(2).sum(-1).mean()

    # ------------------------------------------------------------------
    # h_smooth part (b): the fluid force may only change slowly in time
    # ------------------------------------------------------------------
    def h_fluid_temporal_smooth(self, fluid_series):
        """Drag is a smooth function of relative wind, and relative wind
        changes slowly; contact impulses are the jumpy thing. Penalizing the
        step-to-step change of the fluid force pushes any rapidly-switching
        compensation (which is what stolen friction looks like at contact
        events) out of the fluid channel.

        Needs consecutive predictions: multistep >= 2. Returns 0 at K=1 (the
        trainer warns once).

        fluid_series: list of (B, 3) total fluid accelerations, one per unroll
        step, in graph (not detached - both ends receive gradient).
        """
        if len(fluid_series) < 2:
            return fluid_series[0].new_zeros(())
        diffs = [((b - a) / self.phi_g).pow(2).sum(-1).mean()
                 for a, b in zip(fluid_series[:-1], fluid_series[1:])]
        return torch.stack(diffs).mean()

    # ------------------------------------------------------------------
    # contact sparsity  (proposal Fig. 1: "sparsity regularization for
    # concentrated forces")
    # ------------------------------------------------------------------
    def h_contact_sparsity(self, phi_contact):
        """L1 on contact force magnitudes: contact should be a few loaded
        points, not a diffuse field. Use with care - it also shrinks the
        legitimate resting normal forces, so keep the weight small."""
        return (phi_contact.norm(dim=-1) / self.phi_g).mean()

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------
    def compute_step_terms(self, phi_contact, c_w, v_node, wall_n,
                           a_fluid_total, alpha_fluid, drag_target, weights):
        """All per-step terms as a dict of RAW (unweighted) scalars. Terms
        whose weight is zero are skipped (no wasted compute)."""
        raws = {}
        if weights.get("w_diss", 0) > 0:
            raws["diss"] = self.h_dissipation(phi_contact, c_w, v_node, wall_n)
        if weights.get("w_fluid_anchor", 0) > 0:
            raws["fluid_anchor"] = self.h_fluid_anchor(a_fluid_total, drag_target)
        if weights.get("w_fluid_torque", 0) > 0:
            raws["fluid_torque"] = self.h_fluid_torque(alpha_fluid)
        if weights.get("w_sparse", 0) > 0:
            raws["sparse"] = self.h_contact_sparsity(phi_contact)
        return raws

    @staticmethod
    def weighted_total(raws, weights):
        """sum_j gamma_j h_j  (proposal Eq. 6). raws holds RAW magnitudes;
        weights maps 'w_<name>' -> gamma."""
        total = 0.0
        for name, val in raws.items():
            total = total + weights.get("w_" + name, 0.0) * val
        return total
