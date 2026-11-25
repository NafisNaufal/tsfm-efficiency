"""Uniform zero-shot inference for pretrained TSFMs.

Usage: python run_tsfm.py --model NAME --ctx contexts.npy --H 8 --out out.npz [--batch 256]
contexts.npy: float32 array (N, T) (all contexts of equal length T in one call).
Output npz: mean (N,H), median (N,H), q (N,H,9) for levels 0.1..0.9, plus metadata.
Every model is used zero-shot with its documented default inference settings.
"""
import argparse, json, time, numpy as np, torch

QL = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def _chronos(name, ctx, H, bs):
    from chronos import BaseChronosPipeline
    repo = {"chronos-bolt-small": "amazon/chronos-bolt-small", "chronos-bolt-base": "amazon/chronos-bolt-base",
            "chronos-2": "amazon/chronos-2"}[name]
    p = BaseChronosPipeline.from_pretrained(repo, device_map="cuda", torch_dtype=torch.float32)
    qs, ms = [], []
    for i in range(0, len(ctx), bs):
        x = torch.tensor(ctx[i:i + bs], dtype=torch.float32)
        if name == "chronos-2":
            q, m = p.predict_quantiles(list(x[:, None, :]), prediction_length=H, quantile_levels=QL)
            q = torch.stack([t[0] for t in q]); m = torch.stack([t[0] for t in m])
        else:
            q, m = p.predict_quantiles(x, prediction_length=H, quantile_levels=QL)
        qs.append(q.float().cpu().numpy()); ms.append(m.float().cpu().numpy())
    q = np.concatenate(qs); m = np.concatenate(ms)
    # Chronos(-Bolt/-2) "mean" output is the median quantile; we report it as both and flag it.
    return {"mean": m, "median": q[..., 4], "q": q, "mean_is_median": True}


def _timesfm25(ctx, H, bs):
    import timesfm
    m = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    m.compile(timesfm.ForecastConfig(max_context=max(1024, ctx.shape[1]), max_horizon=max(128, H), normalize_inputs=True,
                                     use_continuous_quantile_head=True, fix_quantile_crossing=True, per_core_batch_size=bs))
    point, qf = m.forecast(horizon=H, inputs=[c for c in ctx])
    qf = np.asarray(qf)       # (N, H, 10): mean + 9 deciles per TimesFM docs
    return {"mean": np.asarray(point)[:, :H], "median": qf[:, :H, 5], "q": qf[:, :H, 1:10], "mean_is_median": False}


def _timesfm3(ctx, H, bs):
    import timesfm
    f = timesfm.TimesFM3Forecaster.from_pretrained("google/timesfm-3.0-pytorch", device="cuda")
    outs = list(f.predict_batch([c for c in ctx], horizon=H, return_quantiles=True))
    point = np.stack([np.asarray(o.forecast)[:H] for o in outs])
    q = np.stack([np.asarray(o.quantiles)[:H] for o in outs])
    if q.shape[-1] == 10: q = q[..., 1:]
    return {"mean": point, "median": q[..., 4], "q": q, "mean_is_median": True}


def _tirex(ctx, H, bs):
    from tirex import load_model
    m = load_model("NX-AI/TiRex", device="cuda")
    q, mean = m.forecast(context=torch.tensor(ctx, dtype=torch.float32), prediction_length=H, batch_size=bs)
    q = q.float().cpu().numpy(); mean = mean.float().cpu().numpy()
    return {"mean": mean, "median": q[..., 4], "q": q, "mean_is_median": True}


def _moirai2(ctx, H, bs, size="small"):
    from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
    T = ctx.shape[1]
    model = Moirai2Forecast(module=Moirai2Module.from_pretrained(f"Salesforce/moirai-2.0-R-{size}"), prediction_length=H,
                            context_length=T, target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0).cuda().eval()
    qs = []
    with torch.no_grad():
        for i in range(0, len(ctx), bs):
            x = torch.tensor(ctx[i:i + bs], dtype=torch.float32, device="cuda")[..., None]
            obs = torch.ones_like(x, dtype=torch.bool); pad = torch.zeros(x.shape[:2], dtype=torch.bool, device="cuda")
            out = model(past_target=x, past_observed_target=obs, past_is_pad=pad)   # (B, n_quantiles, H)
            qs.append(out.float().cpu().numpy())
    q = np.concatenate(qs).transpose(0, 2, 1)        # (N, H, 9)
    return {"mean": q[..., 4], "median": q[..., 4], "q": q, "mean_is_median": True}


def _toto(ctx, H, bs, n_samples=256):
    from toto.model.toto import Toto
    from toto.inference.forecaster import TotoForecaster
    from toto.data.util.dataset import MaskedTimeseries
    toto = Toto.from_pretrained("Datadog/Toto-Open-Base-1.0").cuda().eval()
    fc = TotoForecaster(toto.model)
    means, qs = [], []
    # Toto's space-wise attention folds batch x samples x patches into one CUDA grid dim (limit 65535):
    # keep B * samples_per_batch * n_patches below ~49k. Only throughput changes, not the forecast distribution.
    npatch = ctx.shape[1] // 16 + 2
    bs = max(1, min(bs, 49152 // (8 * npatch)))
    for i in range(0, len(ctx), bs):
        x = torch.tensor(ctx[i:i + bs], dtype=torch.float32, device="cuda")[:, None, :]   # (B, 1 var, T)
        B, _, T = x.shape
        inp = MaskedTimeseries(series=x, padding_mask=torch.ones_like(x, dtype=torch.bool), id_mask=torch.zeros_like(x),
                               timestamp_seconds=torch.zeros_like(x), time_interval_seconds=torch.full((B, 1), 60, device="cuda"))
        spb = max(1, min(n_samples, 49152 // (B * npatch))); spb = 1 << (spb.bit_length() - 1)   # power of 2 divides 256
        f = fc.forecast(inp, prediction_length=H, num_samples=n_samples, samples_per_batch=spb)
        s = f.samples[:, 0].float().cpu().numpy()       # (B, H, S)
        means.append(s.mean(-1)); qs.append(np.quantile(s, QL, axis=-1).transpose(1, 2, 0))
    q = np.concatenate(qs)
    return {"mean": np.concatenate(means), "median": q[..., 4], "q": q, "mean_is_median": False}


def run(name, ctx, H, bs):
    if name.startswith("chronos"): return _chronos(name, ctx, H, bs)
    if name == "timesfm-2.5": return _timesfm25(ctx, H, bs)
    if name == "timesfm-3": return _timesfm3(ctx, H, bs)
    if name == "tirex": return _tirex(ctx, H, bs)
    if name == "moirai-2-small": return _moirai2(ctx, H, bs, "small")
    if name == "toto": return _toto(ctx, H, bs)
    raise KeyError(name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model"); ap.add_argument("--ctx"); ap.add_argument("--H", type=int); ap.add_argument("--out")
    ap.add_argument("--batch", type=int, default=256)
    a = ap.parse_args()
    ctx = np.load(a.ctx).astype(np.float32)
    t0 = time.time(); r = run(a.model, ctx, a.H, a.batch)
    import os
    tmp = a.out[:-4] + f".tmp{os.getpid()}.npz"   # atomic write: parallel workers never see partial files
    np.savez(tmp, mean=r["mean"].astype(np.float32), median=r["median"].astype(np.float32), q=r["q"].astype(np.float32),
             meta=json.dumps({"model": a.model, "mean_is_median": r["mean_is_median"], "N": len(ctx), "T": ctx.shape[1],
                              "H": a.H, "seconds": time.time() - t0}))
    os.replace(tmp, a.out)
    print(a.model, ctx.shape, "->", r["q"].shape, f"{time.time() - t0:.1f}s")
