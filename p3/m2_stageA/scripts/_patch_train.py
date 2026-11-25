# -*- coding: utf-8 -*-
import sys
p = '/data/disk1/home/ghe270/p3/m2_stageA/scripts/train_stageA.py'
s = open(p, encoding='utf-8').read()

# ---- args: residual modes + w-pix + lr-d --------------------------------
old = '''    ap.add_argument("--residual-mode", default="raw", choices=["raw", "x0"])'''
new = '''    ap.add_argument("--residual-mode", default="raw",
                    choices=["raw", "neg_raw", "x0"])'''
assert old in s; s = s.replace(old, new)

old = '''    ap.add_argument("--w-gate", type=float, default=0.01)'''
new = '''    ap.add_argument("--w-gate", type=float, default=0.01)
    ap.add_argument("--w-pix", type=float, default=1.0,
                    help="weight of the Charbonnier term; lower it so the "
                         "residual stops converging to the mean solution")
    ap.add_argument("--lr-d", type=float, default=1e-4,
                    help="discriminator lr; SN-UNet diverges at the generator lr")'''
assert old in s; s = s.replace(old, new)

# ---- discriminator lr ---------------------------------------------------
old = '''        opt_d = torch.optim.AdamW(disc.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)'''
new = '''        # The generator lr (1e-3 in the Pilot) destroys the SN-UNet D within a
        # few dozen steps, so D gets its own, lower rate.
        opt_d = torch.optim.AdamW(disc.parameters(), lr=args.lr_d,
                                  weight_decay=args.weight_decay)'''
assert old in s; s = s.replace(old, new)

# ---- fix 5: gate regulariser + pixel weight ----------------------------
old = '''            # Gate regulariser, two-sided on purpose.
            #
            # The guide writes 0.01 * mean(W_pix * (1 - U_1)), which only ever
            # rewards a *small* W_pix: with no lower bound the cheapest way to
            # satisfy it is W_pix -> 0, and that is exactly what happened
            # (measured: W_pix 0.119 -> 0.000 and w_loss -> 0 within 50 steps,
            # i.e. the model silently became plain RealBasicVSR).  A small
            # anti-collapse term keeps the gate alive; the flat-region term
            # still discourages edits where U_1 is low.
            l_flat = (w_pix * (1.0 - u1)).mean()
            l_alive = ((1.0 - w_pix) ** 2).mean()
            l_gate = args.w_gate * (l_flat + args.gate_alive * l_alive)

            loss = l_pix + args.w_dists * l_dists + args.w_temp * l_temp + l_gate'''
new = '''            # Gate regulariser: anchor the gate to u1 itself.
            #
            # `0.01 * mean(W_pix * (1 - U_1))` (the guide) has its minimum at
            # W_pix -> 0 and collapsed the gate in 50 steps.  The replacement
            # `((1 - W_pix) ** 2)` fixed the collapse but has its analytic
            # minimum at W_pix = 0.50 for EVERY u1 -- which is why
            # gate_vis_p2.png shows flat walls fully lit (~0.6).  Anchoring to
            # u1 makes flat regions (u1 ~ 0.04) drive w -> 0 and texture
            # (u1 ~ 0.39) keep it open, which is the Direction C behaviour the
            # gate is supposed to learn.
            l_gate = args.w_gate * ((w_pix - u1) ** 2).mean()

            # Pixel weight is a knob: at 1.0 the L1 term dominates (only 0.05 of
            # DISTS and no GAN in the Pilot) and pulls a GAN-sharp base toward
            # the mean, which is exactly the LPIPS regression that was measured.
            loss = (args.w_pix * l_pix
                    + args.w_dists * l_dists
                    + args.w_temp * l_temp
                    + l_gate)'''
assert old in s; s = s.replace(old, new)

# ---- fix 6: RaGAN generator/discriminator separation -------------------
old = '''            if disc is not None:
                real_l = disc(gt_c.float())
                fake_l = disc(y_out.float())
                loss = loss + args.w_adv * ra_g_loss(real_l, fake_l)
                dl = ra_d_loss(real_l.detach(), fake_l.detach())
                (dl / accum).backward()
            (loss / accum).backward()'''
new = '''            l_adv_g = torch.zeros((), device=dev)
            l_adv_d = torch.zeros((), device=dev)
            if disc is not None:
                # Generator side: freeze D's parameters so this branch only
                # writes grads into G.  The old code computed a D loss from
                # \`real_l.detach()\` / \`fake_l.detach()\` -- both already
                # detached, so backward() reached nothing and would raise
                # "element 0 of tensors does not require grad" the moment
                # --w-adv > 0.  The Pilot never enabled the GAN, which is why
                # it never surfaced.
                for p_ in disc.parameters():
                    p_.requires_grad_(False)
                real_l = disc(gt_c.float()).detach()
                fake_l = disc(y_out.float())
                l_adv_g = ra_g_loss(real_l, fake_l)
                loss = loss + args.w_adv * l_adv_g

            (loss / accum).backward()

            if disc is not None:
                # Discriminator side: detach the generated image so the gradient
                # only flows through D.
                for p_ in disc.parameters():
                    p_.requires_grad_(True)
                real_d = disc(gt_c.float())
                fake_d = disc(y_out.detach().float())
                l_adv_d = ra_d_loss(real_d, fake_d)
                (l_adv_d / accum).backward()'''
assert old in s; s = s.replace(old, new)

# ---- log adv terms -----------------------------------------------------
old = '''            for k, v in (("loss", loss), ("pix", l_pix), ("dists", l_dists),
                         ("temp", l_temp), ("gate", l_gate)):'''
new = '''            for k, v in (("loss", loss), ("pix", l_pix), ("dists", l_dists),
                         ("temp", l_temp), ("gate", l_gate),
                         ("adv_g", l_adv_g), ("adv_d", l_adv_d)):'''
assert old in s; s = s.replace(old, new)

# ---- D lr follows the cosine schedule too ------------------------------
old = '''        opt.step()
        if disc is not None:
            opt_d.step()'''
new = '''        opt.step()
        if disc is not None:
            if opt_d.param_groups and opt_d.param_groups[0].get("initial_lr") is None:
                for g in opt_d.param_groups:
                    g["initial_lr"] = g["lr"]
            for g in opt_d.param_groups:
                g["lr"] = g["initial_lr"] * (cur_lr / max(1e-12, args.lr))
            opt_d.step()'''
assert old in s; s = s.replace(old, new)

open(p, 'w', encoding='utf-8').write(s)
print("train_stageA.py patched (fixes 5-6)")
