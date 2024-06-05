# Usage: ./scripts/categorize_papers.sh
set -o errexit

fields=(
    "Audio"
    "Computer Vision"
    "Generative Models"
    "Graph Neural Network"
    "Medical"
    "Multimodal"
    "Natural Language Processing"
    "Neuroscience"
    "Recommendation System"
    "Reinforcement Learning"
    "Robotics"
)
for field in "${fields[@]}"
do
    >&2 echo "$field"
    grep -i "$field" data/cache/arxiv/*.txt 2>/dev/null | \
        cut -d":" -f1 | \
        sort -u | \
        grep -v "^Binary file" >data/"$field"_papers.txt
done | sort
