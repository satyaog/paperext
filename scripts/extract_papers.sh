# Usage: # ./scripts/extract_papers.sh PATH/TO/papers.tar.gz
set -o errexit

mkdir -p data/
tar -xzf $1 -C data/
mv data/Users/olivier/paperoni/dev/cache data/
