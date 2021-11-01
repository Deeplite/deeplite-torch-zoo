---
name: Release template
about: Release v Major.Minor.Patch
title: Release v Major.Minor.Patch
labels: ''
assignees: yasseridris, ahmed-deeplite, wernerchaodeeplite

---

- [ ] PR opened
- [ ] Unit and Functional tests pass
- [ ] PR reviewed and merged
- [ ] version bumped in setup.py (and pushed to master)
- [ ] python release.py stage
- [ ] Integration tests pass(overnight)
- [ ] Acceptance tests pass(overnight)
- [ ] Update docs (mention changes in the comments of the release issue)
- [ ] python release.py release
