"""
Curriculum definition for Computer Science & Data Science education.
Provides topic definitions, IDs, metadata, and difficulty levels.
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class Topic:
    topic_id: str
    name: str
    description: str
    category: str


TOPICS: List[Topic] = [
    Topic(
        topic_id="python_basics",
        name="Python Fundamentals",
        description="Core language syntax, data types, list comprehensions, generators, and control flow in Python.",
        category="Programming",
    ),
    Topic(
        topic_id="data_structures",
        name="Data Structures",
        description="Arrays, linked lists, stacks, queues, hash tables, heaps, binary search trees, and graphs.",
        category="Computer Science",
    ),
    Topic(
        topic_id="algorithms",
        name="Algorithms & Complexity",
        description="Big-O notation, sorting, binary search, divide and conquer, greedy methods, and dynamic programming.",
        category="Computer Science",
    ),
    Topic(
        topic_id="linear_algebra",
        name="Linear Algebra",
        description="Vector spaces, matrix multiplication, determinants, eigenvalues, eigenvectors, and singular value decomposition.",
        category="Mathematics",
    ),
    Topic(
        topic_id="probability_stats",
        name="Probability & Statistics",
        description="Random variables, conditional probability, Bayes theorem, distributions, central limit theorem, hypothesis testing.",
        category="Mathematics",
    ),
    Topic(
        topic_id="machine_learning",
        name="Machine Learning Foundations",
        description="Linear/logistic regression, decision trees, support vector machines, clustering, bias-variance tradeoff, cross-validation.",
        category="Artificial Intelligence",
    ),
    Topic(
        topic_id="deep_learning",
        name="Deep Learning & Neural Networks",
        description="Perceptrons, backpropagation, activation functions, CNNs, sequence models, self-attention, and PyTorch essentials.",
        category="Artificial Intelligence",
    ),
    Topic(
        topic_id="databases_sql",
        name="Relational Databases & SQL",
        description="Relational data modeling, SQL queries, joins, aggregations, indexing, transactions, ACID properties, and normalization.",
        category="Software Engineering",
    ),
    Topic(
        topic_id="web_development",
        name="Web Architecture & APIs",
        description="Client-server architecture, HTTP protocol, REST APIs, JSON serialization, asynchronous endpoints, and backend services.",
        category="Software Engineering",
    ),
    Topic(
        topic_id="system_design",
        name="System Design & Distributed Systems",
        description="High availability, horizontal scaling, caching strategies (Redis), message queues, load balancing, and microservices.",
        category="Software Engineering",
    ),
]

TOPIC_DICT: Dict[str, Topic] = {t.topic_id: t for t in TOPICS}
TOPIC_IDS: List[str] = [t.topic_id for t in TOPICS]
FORMATS: List[str] = ["video", "reading", "interactive", "practice"]
DIFFICULTIES: List[str] = ["beginner", "intermediate", "advanced"]
