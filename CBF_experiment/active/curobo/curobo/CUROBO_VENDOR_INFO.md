# cuRobo Vendored Source

This directory contains a vendored copy of NVIDIA cuRobo, merged into the
parent project `DifferentiableOptimizationCBF-main-master`.

## Upstream

- Repository: https://github.com/NVlabs/curobo
- Vendored commit: `d64c4b005459db10c5dd867d8b30a87d5bda9bdb`
- Author: Balakumar Sundaralingam <s.balakumar@outlook.com>
- Date: 2026-03-02 09:27:47 -0800
- Subject: Merge pull request #587 from WYYAHYT/main

## Local Modifications

- `setup.py`: Added unconditional `--allow-unsupported-compiler` flag to nvcc
  so cuRobo can be compiled against the higher-version GCC shipped in our
  conda environment (originally limited to `sys.platform == "win32"`).

## How to Sync With Upstream

To pull a newer cuRobo release into this folder:

```bash
# from any temporary directory
git clone --depth 1 --branch <tag-or-branch> https://github.com/NVlabs/curobo.git curobo_new
rm -rf curobo_new/.git
# diff and copy what you need over the current folder, then re-apply local mods
```

After updating, refresh this file with the new commit hash.
