# -*- coding: utf-8 -*-
import sys
p = '/data/disk1/home/ghe270/p3/m2_stageA/scripts/model_stageA.py'
s = open(p, encoding='utf-8').read()

# ---------------------------------------------------------------- fix 1+2
old1 = '''        H, W = f_fwd.shape[-2:]
        gx, gy = torch.meshgrid(
            torch.arange(H, device=self.device, dtype=f_fwd.dtype),
            torch.arange(W, device=self.device, dtype=f_fwd.dtype),
            indexing="ij")
        bx = (gx[None, None] + f_fwd[:, 0:1]).clamp(0, W - 1)
        by = (gy[None, None] + f_fwd[:, 1:2]).clamp(0, H - 1)
        nx = (bx / max(W - 1, 1)) * 2 - 1
        ny = (by / max(H - 1, 1)) * 2 - 1
        grid = torch.cat([nx, ny], dim=1).permute(0, 2, 3, 1)
        b_back = F.grid_sample(f_bwd, grid, mode="bilinear",
                               padding_mode="border", align_corners=True)
        fb = (f_fwd + b_back).abs().sum(dim=1, keepdim=True)   # sum, see above
        occ = (fb <= 3.0).to(f_fwd.dtype)
        # centre crop only when the LR input is the training crop size
        if W > HR_OUT // 4:
            sl = slice(BR, BR + HR_OUT // 4)
            return f_fwd[:, :, sl, sl], occ[:, :, sl, sl]
        return f_fwd, occ'''
new1 = '''        H, W = f_fwd.shape[-2:]
        # `indexing="ij"` returns (rows, cols): the FIRST value walks dim 0
        # (the y axis).  Binding it to a variable called gx and adding it to
        # f_fwd[:, 0] (the x channel) transposed the warp 90 degrees.  Verified
        # before the fix on a 96x96 crop: no-warp |B_t - B_prev| = 0.1444 but
        # the buggy warp gave 0.1977, i.e. motion compensation made the
        # frame difference *worse*, and U_warp / L_temp were computed against a
        # transposed neighbour the whole time.
        gy, gx = torch.meshgrid(
            torch.arange(H, device=self.device, dtype=f_fwd.dtype),
            torch.arange(W, device=self.device, dtype=f_fwd.dtype),
            indexing="ij")
        bx = (gx[None, None] + f_fwd[:, 0:1]).clamp(0, W - 1)
        by = (gy[None, None] + f_fwd[:, 1:2]).clamp(0, H - 1)
        nx = (bx / max(W - 1, 1)) * 2 - 1
        ny = (by / max(H - 1, 1)) * 2 - 1
        grid = torch.cat([nx, ny], dim=1).permute(0, 2, 3, 1)
        b_back = F.grid_sample(f_bwd, grid, mode="bilinear",
                               padding_mode="border", align_corners=True)
        fb = (f_fwd + b_back).abs().sum(dim=1, keepdim=True)   # sum, see above
        occ = (fb <= 3.0).to(f_fwd.dtype)
        # Centre-crop ONLY for the 96x96 training crop.  The old guard was
        # `if W > HR_OUT // 4` (i.e. W > 64), which is also true for the
        # 96x96 training crop AND for the 320x180 inference frame: training
        # silently used the top-left 64x64 of its own flow field, and
        # full-frame inference cropped the whole frame's flow to 64x64 and then
        # stretched it over 720p.
        if H == LR_CROP and W == LR_CROP:
            sl = slice(BR, BR + HR_OUT // 4)
            return f_fwd[:, :, sl, sl], occ[:, :, sl, sl]
        return f_fwd, occ'''
assert old1 in s, "fix1 pattern not found"
s = s.replace(old1, new1)

# ---------------------------------------------------------------- fix 3
old2 = '''        b, _, h, w = x.shape
        s = h / flow_lr_c.shape[-1]               # 4 at train, 4 at inference
        # The grid must carry x's dtype: grid_sample requires input and grid to
        # match.  arange defaults to int64, so the cast has to be explicit.
        f = F.interpolate(flow_lr_c.to(x.dtype), size=(h, w),
                          mode="bilinear", align_corners=False) * s
        # (H, W) order: meshgrid(arange(h), arange(w)) -- see the note in
        # _flow_and_occ; passing w first only worked for square crops.
        gx, gy = torch.meshgrid(
            torch.arange(h, device=x.device, dtype=x.dtype),
            torch.arange(w, device=x.device, dtype=x.dtype), indexing="ij")
        px = (gx[None, None] + f[:, 0:1]).clamp(0, w - 1)
        py = (gy[None, None] + f[:, 1:2]).clamp(0, h - 1)
        nx = (px / (w - 1)) * 2 - 1
        ny = (py / (h - 1)) * 2 - 1
        grid = torch.cat([nx, ny], dim=1).permute(0, 2, 3, 1)'''
new2 = '''        b, _, h, w = x.shape
        # Separate x and y scales.  The old single `s = h / flow_w` divided the
        # HEIGHT by the flow WIDTH: 4.0 for a square crop (right by luck) but
        # 720/320 = 2.25 on a full frame, and 720/64 = 11.25 once the crop bug
        # above was combined with it.
        sx = float(w) / float(flow_lr_c.shape[-1])
        sy = float(h) / float(flow_lr_c.shape[-2])
        f = F.interpolate(flow_lr_c.to(x.dtype), size=(h, w),
                          mode="bilinear", align_corners=False)
        f = torch.cat([f[:, 0:1] * sx, f[:, 1:2] * sy], dim=1)
        # `indexing="ij"` -> (rows, cols) = (y, x); see _flow_and_occ.
        gy, gx = torch.meshgrid(
            torch.arange(h, device=x.device, dtype=x.dtype),
            torch.arange(w, device=x.device, dtype=x.dtype), indexing="ij")
        px = (gx[None, None] + f[:, 0:1]).clamp(0, w - 1)
        py = (gy[None, None] + f[:, 1:2]).clamp(0, h - 1)
        nx = (px / max(w - 1, 1)) * 2 - 1
        ny = (py / max(h - 1, 1)) * 2 - 1
        grid = torch.cat([nx, ny], dim=1).permute(0, 2, 3, 1)'''
assert old2 in s, "fix3 pattern not found"
s = s.replace(old2, new2)

# ---------------------------------------------------------------- fix 4
old3 = '''        dZ = self.unet(z_base, torch.tensor(999, device=self.device),
                       encoder_hidden_states=self._pooled_emb(b)).sample
        if self.residual_mode == "x0":
            ab = self.vae_alpha_bar(z_base)
            x0 = (z_base - (1 - ab).sqrt() * dZ) / ab.sqrt()
            dZ = x0 - z_base

        if gate_mode == "learn":
            w_pix, m_lat = self.gate(torch.cat([b_t.to(self.dtype), u1, uw], 1))
        elif gate_mode == "const":
            w_pix = torch.full((b, 1, h, w), float(fixed_w),
                               device=self.device, dtype=self.dtype)
            m_lat = F.avg_pool2d(w_pix, 8, 8, ceil_mode=True)
        elif gate_mode == "u1":
            w_pix = (1.0 - u1).to(self.dtype)
            m_lat = F.avg_pool2d(w_pix, 8, 8, ceil_mode=True)
        else:
            raise ValueError(gate_mode)'''
new3 = '''        dZ = self.unet(z_base, torch.tensor(999, device=self.device),
                       encoder_hidden_states=self._pooled_emb(b)).sample
        if self.residual_mode == "neg_raw":
            # osediff.pkl predicts the noise eps.  With a positive gamma the
            # noise direction sharpens the image once subtracted, so flipping
            # the sign lets gamma stay positive for the P3/P4 arms instead of
            # drifting to -0.45 as it did when "raw" was used with osediff
            # init (measured in the M2 Pilot).
            dZ = -dZ
        elif self.residual_mode == "x0":
            ab = self.vae_alpha_bar(z_base)
            x0 = (z_base - (1 - ab).sqrt() * dZ) / ab.sqrt()
            dZ = x0 - z_base

        if gate_mode == "learn":
            w_pix, m_lat = self.gate(torch.cat([b_t.to(self.dtype), u1, uw], 1))
        elif gate_mode == "const":
            w_pix = torch.full((b, 1, h, w), float(fixed_w),
                               device=self.device, dtype=self.dtype)
            m_lat = F.avg_pool2d(w_pix, 8, 8, ceil_mode=True)
        elif gate_mode == "u1":
            # Direction C: high uncertainty (texture) -> open the gate, flat
            # regions -> close it.  The old `1.0 - u1` was inverted: measured
            # u1 sits at 0.04-0.39, so it drove the gate to 0.61-0.96 on flat
            # regions and closed it on texture, which is what blew P1's E_warp
            # up by +30.9%.
            w_pix = u1.to(self.dtype)
            m_lat = F.avg_pool2d(w_pix, 8, 8, ceil_mode=True)
        else:
            raise ValueError(gate_mode)'''
assert old3 in s, "fix4 pattern not found"
s = s.replace(old3, new3)

open(p, 'w', encoding='utf-8').write(s)
print("model_stageA.py patched (fixes 1-4)")
