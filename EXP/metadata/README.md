# Recovered run metadata

Each implementation folder has two inputs absent from the original EXP snapshot:

- `s1.npy`: the matching secret polynomial used to score recovered key coefficients.
- `seed_and_rej_num.txt`: 10,000 original seeds and retained-candidate counts used to group rejected and valid signatures.

The M4 pair was copied from the supplied `mldsa_m4旧实现-168MHz` reference snapshot; the reference pair was copied from `mldsa_ref-O3`. The EXP data and raw traces were spot-checked byte-for-byte against those same snapshots before copying. Keep each metadata pair with its corresponding implementation.
