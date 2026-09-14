## What this is

- [ ] Bug fix
- [ ] New patch
- [ ] Docs / CI only

## Image

Pinned digest you ran against (`sha256:` + 12 or more hex):

```
vllm/vllm-openai-xpu@sha256:
```

Family (one): Qwen3.8 / Qwen3.6 / Nemotron / other

Prefix cache: on / off
TP: 1 / 2 / 4

## What you ran

Paste the command and the last line of output. GPU-free `docker run` without `--device` is enough for text patches.

```
```

## What you are not claiming

- [ ] No live corruption rate (or paste n= and the count)
- [ ] Apply-list does not mix Qwen MTP patches onto Nemotron or the reverse

## Authority and validation

Owning authority changed (one): system map / benchmark catalog / benchmark
contract / image-patch matrix / reliability map / family recipe

- [ ] I followed `docs/ADDING-A-RECIPE.md` for a new or changed route
- [ ] `python3 scripts/check-repo.py`
- [ ] Generated views were regenerated, not hand-edited
