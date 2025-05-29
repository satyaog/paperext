import json
import csv
from collections import Counter
from statistics import mean


def write_paper_topics_csv(input_json_path, output_csv_path):
    # Read JSON file
    with open(input_json_path, "r") as f:
        papers = json.load(f)

    # Write CSV file
    with open(output_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(["Title", "Paper ID", "Topics"])

        # Write data for each paper
        for paper in papers:
            title = paper["title"]
            paper_id = paper["paper_id"]
            for topic in paper["topics"]:
                writer.writerow([title, paper_id, topic["name"]])


def analyze_paper_topics(input_json_path):
    # Read JSON file
    with open(input_json_path, "r") as f:
        papers = json.load(f)

    # Initialize counters and lists
    topic_counter = Counter()
    topics_per_paper = []
    papers_without_topics = 0

    # Analyze each paper
    for paper in papers:
        topics = paper["topics"]
        if not topics:
            papers_without_topics += 1
            topics_per_paper.append(0)
        else:
            topics_per_paper.append(len(topics))
            for topic in topics:
                topic_counter[topic["name"].lower()] += 1

    # Calculate statistics
    total_papers = len(papers)
    avg_topics = mean(topics_per_paper) if topics_per_paper else 0

    # Print statistics
    print(f"\nPaper Topics Analysis:")
    print(f"Total number of papers: {total_papers}")
    print(f"Number of topics: {len(topic_counter)}")
    print(f"Average topics per paper: {avg_topics:.2f}")
    print(f"Papers without topics: {papers_without_topics}")
    print(f"\nTop 100 most common topics:")
    for topic, count in topic_counter.most_common(100):
        print(f"{topic}: {count} papers")


if __name__ == "__main__":
    input_path = "data/paperoni---PR_2025-04-03.json"
    write_paper_topics_csv(input_path, "papers_topics.csv")
    analyze_paper_topics(input_path)
