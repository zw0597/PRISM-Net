# R9 post-analysis (after R9; stage chosen on the validation radiographs V): CAA alone vs the paper's chain, recalibration for CAA.
import numpy as np, json
exec(open("r9_chain.py").read().split('if __name__ == "__main__":')[0])
VV = np.isin(IMG, SP["V"]); P = np.load("swrd/swrd_probs_r9.npz"); R8 = np.load("swrd/swrd_probs_r8.npz"); S9 = json.load(open("r9_summary.json"))
names = MODELS + ["gpu:comp", "gpu:oc"]; out = {}
for m in names:
    corr = lambda A: np.mean([(A[f].argmax(1) == YS) for f in range(5)], 0)
    o = {s: S9[m][s] for s in ("pgaR", "pgaR_bn", "caa", "caa_bn", "caa_bn_up")}
    o["val_acc"] = {s: float(np.mean([(P[f"{m}|{s}"][f][VV].argmax(1) == YS[VV]).mean() for f in range(5)])) for s in ("pgaR", "pgaR_bn", "caa", "caa_bn", "caa_bn_up")}
    o["caa-pgaR_bn"] = rad_boot_diff(corr(P[f"{m}|caa"]), corr(P[f"{m}|pgaR_bn"]))
    o["recal_caa"] = recal(list(P[f"{m}|caa"])); o["recal_pgaR_bn"] = S9[m]["recal"]["pgaR_bn"]; o["recal_caa_bn_up"] = S9[m]["recal"]["caa_bn_up"]
    if m in MODELS:
        A = R8[f"{m}|D"]; o["caaP"] = {"acc": float(np.mean([(A[f][TE].argmax(1) == YS[TE]).mean() for f in range(5)])), "ece": float(np.mean([ece(A[f][TE], YS[TE]) for f in range(5)]))}
    out[m] = o
json.dump(out, open("r9_post.json", "w"), indent=1)
for m in names:
    o = out[m]; d = o["caa-pgaR_bn"]
    print(f"{m:24s} CAA {o['caa']['acc']*100:5.1f} vs PGA-R+BN {o['pgaR_bn']['acc']*100:5.1f}: {d['d']*100:+.1f} [{d['lo']*100:.1f},{d['hi']*100:.1f}] | ECE CAA {o['caa']['ece']*100:4.1f} UP {o['caa_bn_up']['ece']*100:4.1f} | single CAA {o['recal_caa']['single']*100:4.1f} PGA {o['recal_pgaR_bn']['single']*100:4.1f} cov {o['recal_caa']['cov']*100:4.1f}")
