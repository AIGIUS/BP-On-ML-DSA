# Experimental values

`mldsa_m4_data/` and `mldsa_ref_data/` contain 10,000 valid `y` and `z` vectors and the candidate `c`/`cs1` vectors for the matching implementation. Reference `z_valid/` was derived from `y+cs1`; the derivation is reproducible with `python ../py_code/reproduce.py derive-z --profile ref`. Candidate indices follow `seed_and_rej_num.txt` in `../metadata/`. See the [archive README](../README.md) for the data layout and commands.
