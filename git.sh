set -e

# Default commit message
MESSAGE="${1:-Update}"

# Add all changes, excluding specific files/directories
git add --ignore-errors . || true
git reset -- core verl-agent-exp-latest.tar 2>/dev/null || true

# Commit with message
git commit -m "$MESSAGE"

# Push to remote
git push verl-agent-exp main

echo "Successfully pushed to verl-agent-exp"