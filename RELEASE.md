# Releasing FastEmbed

This is a guide how to release `fastembed` and `fastembed-gpu` packages.

## How to

1. Accumulate changes in the `main` branch.
2. Bump the version in `pyproject.toml`

3. Rebase the `gpu` branch on `main` and resolve conflicts if occurred:

```bash
git checkout gpu
git rebase main
git push origin gpu
```

4. Draft release notes
5. Checkout to `main` and create a tag, e.g.:

```bash
git checkout main
git tag -a v0.1.0 -m "Release v0.1.0"
```

6. Checkout `gpu` and create a tag, e.g.:

```bash
git checkout gpu
git tag -a v0.1.0-gpu -m "Release v0.1.0"
```

7. Push tags:

```bash
git push --tags
```

8. Verify that both packages have been published successfully on PyPI. Try installing them and verify imports.
9. Create a release on GitHub with the written release notes.

