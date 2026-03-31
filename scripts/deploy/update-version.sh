#!/usr/bin/env bash
# ============================================================
# SAM3 Version Bump Script
# ─────────────────────────────────────────────────────────────
# Bump version in pyproject.toml and create a git tag.
#
# Usage:
#   ./scripts/deploy/update-version.sh patch   # 1.0.0 → 1.0.1
#   ./scripts/deploy/update-version.sh minor   # 1.0.0 → 1.1.0
#   ./scripts/deploy/update-version.sh major   # 1.0.0 → 2.0.0
# ============================================================

set -euo pipefail

BUMP="${1:-patch}"
MARKER="## planned"
DATE=$(date '+%Y-%m-%d')

if [[ ! "$BUMP" =~ ^(patch|minor|major)$ ]]; then
    echo "Usage: $0 {patch|minor|major}"
    exit 1
fi

# ── Read current version ────────────────────────────────────
CURRENT=$(grep '^version = ' pyproject.toml | head -1 | sed 's/version = "//;s/"//' | tr -d ' ')
echo "Current version: $CURRENT"

# ── Bump ────────────────────────────────────────────────────
MAJOR=$(echo "$CURRENT" | cut -d. -f1)
MINOR=$(echo "$CURRENT" | cut -d. -f2)
PATCH=$(echo "$CURRENT" | cut -d. -f3)

case "$BUMP" in
    major) MAJOR=$((MAJOR + 1)); MINOR=0; PATCH=0 ;;
    minor) MINOR=$((MINOR + 1)); PATCH=0 ;;
    patch) PATCH=$((PATCH + 1)) ;;
esac

NEW="${MAJOR}.${MINOR}.${PATCH}"
echo "New version: $NEW"

# ── Update pyproject.toml ──────────────────────────────────
sed -i "s/^version = \".*\"/version = \"$NEW\"/" pyproject.toml

# ── Update __init__.py or main.py if it has a version ───────
if grep -q '__version__' src/__init__.py 2>/dev/null; then
    sed -i "s/__version__ = \".*\"/__version__ = \"$NEW\"/" src/__init__.py
fi

# ── Update main.py version ─────────────────────────────────
sed -i "s/\"version\": \"$CURRENT\"/\"version\": \"$NEW\"/" main.py

# ── Update CLAUDE.md version ────────────────────────────────
if grep -q "^\\*\\*Version\\*\\*:" .claude/CLAUDE.md 2>/dev/null; then
    sed -i "s/\*\*Version\*\*:.*/\*\*Version\*\*: $NEW/" .claude/CLAUDE.md
fi

# ── Commit & Tag ────────────────────────────────────────────
git add pyproject.toml main.py .claude/CLAUDE.md 2>/dev/null || true
git add src/__init__.py 2>/dev/null || true
git commit -m "chore: bump version to v$NEW"
git tag -a "v$NEW" -m "Release v$NEW ($DATE)"

echo ""
echo "✅ Version bumped to v$NEW"
echo "   Commit: $(git rev-parse --short HEAD)"
echo "   Tag: v$NEW"
echo ""
echo "Push with:"
echo "  git push && git push --tags"
