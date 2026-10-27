"""
Catalog of 300 curated educational resources across Computer Science and Data Science.
Each resource carries:
- Unique resource ID
- Title and detailed description (for TF-IDF feature vector extraction)
- Topic tags (1-2 topics)
- Format (video, reading, interactive, practice)
- Difficulty (beginner, intermediate, advanced)
- Source / Author attribution
- Estimated duration
"""

from dataclasses import dataclass, field
from typing import List, Dict
import random
from jaash_rs.domain.curriculum import TOPIC_IDS, FORMATS, DIFFICULTIES, TOPIC_DICT


@dataclass
class LearningResource:
    resource_id: str
    title: str
    description: str
    topics: List[str]
    format: str
    difficulty: str
    estimated_minutes: int
    source: str = "Academic Learning Consortium"

    def to_dict(self) -> Dict:
        return {
            "resource_id": self.resource_id,
            "title": self.title,
            "description": self.description,
            "topics": self.topics,
            "format": self.format,
            "difficulty": self.difficulty,
            "estimated_minutes": self.estimated_minutes,
            "source": self.source,
        }


# Curated resource blueprints per topic to ensure realistic TF-IDF descriptions
_TOPIC_TEMPLATES = {
    "python_basics": [
        ("Python Variable Scoping, Namespaces, and LEGB Rules",
         "Explore local, enclosing, global, and built-in scopes in Python. Master the global and nonlocal keywords with debugging examples.",
         ["video", "reading"], ["beginner", "intermediate"]),
        ("Data Structures Built into Python: Lists, Tuples, and Dicts",
         "Comprehensive guide on time complexities of internal Python collections, dictionary hashing, and memory footprint comparisons.",
         ["interactive", "reading"], ["beginner", "intermediate"]),
        ("Mastering Python List Comprehensions and Generator Expressions",
         "Write clean, idiomatic Python using list, set, and dictionary comprehensions. Learn memory-efficient stream processing with yield.",
         ["practice", "interactive"], ["beginner", "intermediate"]),
        ("Object-Oriented Programming in Python: Dunder Methods and Inheritance",
         "Learn Python class models, magic dunder methods (__init__, __repr__, __getitem__), multiple inheritance, and method resolution order.",
         ["reading", "video"], ["intermediate", "advanced"]),
        ("Functional Programming in Python: Lambdas, Map, Filter, and Closures",
         "Higher-order functions, functional composition, and closure state encapsulation in clean modern Python code.",
         ["interactive", "practice"], ["intermediate"]),
        ("Python Exception Handling and Custom Error Hierarchies",
         "Best practices for try-except-finally blocks, context managers with 'with' statements, and writing informative exception classes.",
         ["reading", "practice"], ["beginner", "intermediate"]),
        ("Advanced Python Decorators, Metaclasses, and Introspection",
         "Deep dive into parameterized decorators, functools.wraps, class decorators, and runtime class construction with metaclasses.",
         ["reading", "video"], ["advanced"]),
        ("Concurrency in Python: Threading, Multiprocessing, and Asyncio",
         "Understand the Global Interpreter Lock (GIL), cooperative multitasking with async/await event loops, and CPU-bound parallel multiprocessing.",
         ["video", "interactive"], ["advanced"]),
        ("Python Standard Library Essentials: itertools, collections, and pathlib",
         "Accelerate development using Counter, defaultdict, deque, chain, combinations, and modern file operations via Path objects.",
         ["practice", "reading"], ["intermediate"]),
        ("Writing Unit Tests in Python with Pytest and Mocking",
         "Test-driven development fundamentals, parameterized test fixtures, mocking external I/O dependencies, and assertion testing.",
         ["practice", "interactive"], ["intermediate", "advanced"]),
    ],
    "data_structures": [
        ("Dynamic Arrays and Amortized Complexity Analysis",
         "How dynamic array resizing works under the hood. Mathematical proof of O(1) amortized insertion and memory doubling policies.",
         ["reading", "video"], ["beginner", "intermediate"]),
        ("Singly and Doubly Linked Lists: Pointer Manipulation",
         "Implement linked lists from scratch. Master pointer swaps, fast-slow pointer cycle detection (Floyd's algorithm), and reversal.",
         ["interactive", "practice"], ["beginner", "intermediate"]),
        ("Stack and Queue Implementations with Monotonic Optimizations",
         "Build LIFO stacks and FIFO queues. Solve classic problems like balanced parentheses, daily temperatures, and sliding window maximum.",
         ["practice", "video"], ["intermediate"]),
        ("Hash Tables, Collision Handling, and Load Factor Management",
         "Internal mechanics of hash tables: hash functions, open addressing, linear probing, robin hood hashing, and separate chaining.",
         ["interactive", "reading"], ["intermediate"]),
        ("Binary Search Trees: Traversal, Balancing, and Invariants",
         "Inorder, preorder, postorder, and level-order tree traversals. Invariant verification, node insertion, and deletion algorithms.",
         ["video", "practice"], ["intermediate"]),
        ("Self-Balancing AVL Trees and Red-Black Trees",
         "Maintain logarithmic search trees through tree rotations, balance factors, color flips, and height restoration rules.",
         ["reading", "interactive"], ["advanced"]),
        ("Binary Heaps and Priority Queues: Heapify and HeapSort",
         "Min-heap and max-heap array representations. Sift-up and sift-down operations, building heaps in O(N) time, and k-way merging.",
         ["video", "practice"], ["intermediate"]),
        ("Trie Data Structures for Prefix Searching and Autocomplete",
         "Construct prefix trees for fast dictionary lookups, wildcard searching, and longest prefix matching in IP routing.",
         ["interactive", "practice"], ["intermediate", "advanced"]),
        ("Disjoint Set Union (Union-Find) with Path Compression",
         "Near-constant time set union and find operations using rank heuristics and path compression. Applications in Kruskal's algorithm.",
         ["reading", "practice"], ["intermediate", "advanced"]),
        ("Graph Representations: Adjacency Matrices vs Adjacency Lists",
         "Memory and access time tradeoffs for sparse and dense graphs. Storing directed, undirected, and weighted edge topologies.",
         ["video", "interactive"], ["beginner", "intermediate"]),
    ],
    "algorithms": [
        ("Asymptotic Complexity, Big-O, Big-Omega, and Big-Theta",
         "Formal mathematical definitions of algorithmic bounds. Analyzing loops, recursion trees, and worst-case vs average-case behavior.",
         ["reading", "video"], ["beginner", "intermediate"]),
        ("Divide and Conquer: MergeSort and QuickSort In-Depth",
         "Master the Master Theorem, recurrence relations, partition strategies (Lomuto vs Hoare), and dual-pivot QuickSort optimization.",
         ["interactive", "practice"], ["intermediate"]),
        ("Binary Search and Its Variants on Monotonic Solution Spaces",
         "Beyond array lookups: binary search over answer spaces, finding square roots, peak elements, and capacity planning problems.",
         ["practice", "interactive"], ["intermediate"]),
        ("Graph Traversal: Breadth-First Search (BFS) and Shortest Paths",
         "Queue-based BFS for unweighted graphs, multi-source BFS, topological sorting via Kahn's algorithm, and cycle detection.",
         ["video", "practice"], ["beginner", "intermediate"]),
        ("Depth-First Search (DFS), Backtracking, and Recursion",
         "Recursive exploration, cycle detection in directed graphs, connected components, and solving N-Queens and Sudoku.",
         ["interactive", "practice"], ["intermediate"]),
        ("Shortest Path Algorithms: Dijkstra, Bellman-Ford, and Floyd-Warshall",
         "Greedy relaxation with priority queues, negative weight detection, and all-pairs dynamic programming shortest path calculations.",
         ["reading", "video"], ["intermediate", "advanced"]),
        ("Minimum Spanning Trees: Kruskal and Prim Algorithms",
         "Greedy approaches to graph spanning trees, cut property proofs, and runtime comparisons using Fibonacci heaps.",
         ["video", "practice"], ["intermediate"]),
        ("Dynamic Programming: Memoization vs Tabulation Paradigms",
         "Decompose optimal substructure and overlapping subproblems. From Fibonacci to 0/1 Knapsack, coin change, and matrix chain multiplication.",
         ["interactive", "practice"], ["intermediate", "advanced"]),
        ("String Matching Algorithms: KMP and Rabin-Karp Rolling Hash",
         "Linear time pattern searching using failure functions, pi tables, and polynomial rolling hash collisions.",
         ["reading", "video"], ["advanced"]),
        ("Greedy Algorithms, Interval Scheduling, and Huffman Coding",
         "Proving greedy choice property, optimal prefix codes for lossless data compression, and job scheduling with deadlines.",
         ["reading", "practice"], ["intermediate"]),
    ],
    "linear_algebra": [
        ("Vectors, Vector Spaces, Linear Combinations, and Span",
         "Geometric intuition of vectors, basis vectors, linear independence, dimension, and coordinate systems in Euclidean spaces.",
         ["video", "reading"], ["beginner", "intermediate"]),
        ("Matrix Multiplication as Linear Transformations",
         "Geometric interpretation of matrices as space transformations: rotations, scaling, reflections, and shear operations.",
         ["interactive", "video"], ["beginner", "intermediate"]),
        ("Systems of Linear Equations and Gaussian Elimination",
         "Solving linear equations using row reduction, augmented matrices, row echelon form, and rank-nullity theorem.",
         ["practice", "reading"], ["beginner", "intermediate"]),
        ("Matrix Inverse, Determinants, and Orthogonality",
         "Properties of non-singular matrices, geometric meaning of determinant as volume scaling, and orthogonal projection matrices.",
         ["reading", "interactive"], ["intermediate"]),
        ("Eigenvalues and Eigenvectors: Characteristic Equations",
         "Finding invariant axes under transformations. Calculating characteristic polynomials, algebraic vs geometric multiplicity.",
         ["video", "practice"], ["intermediate", "advanced"]),
        ("Matrix Diagonalization and Spectral Theorem for Symmetric Matrices",
         "Decoupling linear transformations with eigen-decomposition: A = PDP^-1. Orthogonal diagonalization of symmetric matrices.",
         ["reading", "interactive"], ["intermediate", "advanced"]),
        ("Singular Value Decomposition (SVD) and Low-Rank Approximations",
         "Decomposing arbitrary rectangular matrices into U, Sigma, V^T. Dimensionality reduction, pseudoinverses, and image compression.",
         ["video", "reading"], ["advanced"]),
        ("Principal Component Analysis (PCA) Linear Algebra Derivation",
         "Maximizing variance and minimizing projection error using covariance matrix eigenvectors. Connecting PCA directly to SVD.",
         ["interactive", "practice"], ["intermediate", "advanced"]),
        ("Vector and Matrix Norms: L1, L2, Frobenius, and Spectral Norms",
         "Measuring vector magnitude and matrix perturbation sensitivities, condition numbers, and matrix stability analysis.",
         ["reading", "practice"], ["intermediate"]),
        ("Positive Semi-Definite Matrices and Quadratic Forms",
         "Ellipsoids in high dimensions, Cholesky decomposition, Hessian matrix concavity in optimization, and kernel methods.",
         ["video", "reading"], ["advanced"]),
    ],
    "probability_stats": [
        ("Foundations of Probability: Sample Spaces, Axioms, and Counting",
         "Permutations, combinations, probability axioms of Kolmogorov, mutually exclusive events, and law of total probability.",
         ["reading", "practice"], ["beginner"]),
        ("Conditional Probability, Independence, and Bayes' Theorem",
         "Prior, likelihood, marginal likelihood, and posterior probability. Disease screening paradox and updating beliefs with evidence.",
         ["interactive", "video"], ["beginner", "intermediate"]),
        ("Discrete Random Variables: Bernoulli, Binomial, and Poisson",
         "Probability mass functions, expected values, variance, and Poisson approximations of rare events in real-world systems.",
         ["practice", "reading"], ["intermediate"]),
        ("Continuous Distributions: Normal, Exponential, and Uniform",
         "Probability density functions, cumulative distribution functions, Gaussian properties, 68-95-99.7 rule, and z-score calculations.",
         ["video", "interactive"], ["beginner", "intermediate"]),
        ("Joint Distributions, Covariance, and Pearson Correlation",
         "Bivariate distributions, marginal distributions, conditional expectation, covariance formulas, and measuring linear dependence.",
         ["interactive", "reading"], ["intermediate"]),
        ("Law of Large Numbers and Central Limit Theorem",
         "Why sample means converge to Gaussian distributions regardless of the underlying population distribution. Simulation experiments.",
         ["interactive", "video"], ["intermediate"]),
        ("Maximum Likelihood Estimation (MLE) and MAP Principles",
         "Formulating log-likelihood functions, finding parameter argmax via derivatives, and regularized estimation with Bayesian priors.",
         ["reading", "practice"], ["intermediate", "advanced"]),
        ("Confidence Intervals and Hypothesis Testing (p-values)",
         "Null and alternative hypotheses, Type I and Type II errors, statistical power, two-tailed t-tests, and p-value interpretations.",
         ["practice", "video"], ["intermediate"]),
        ("Chi-Square Tests and ANOVA: Comparing Multiple Groups",
         "Goodness-of-fit tests, contingency table independence checks, and one-way Analysis of Variance F-test statistics.",
         ["reading", "practice"], ["intermediate", "advanced"]),
        ("Bayesian Inference, Conjugate Priors, and Beta-Binomial Updating",
         "Deriving analytical posterior distributions using conjugate priors. Beta distribution updating for coin flips and success rates.",
         ["interactive", "reading"], ["advanced"]),
    ],
    "machine_learning": [
        ("Linear Regression with Ordinary Least Squares and Gradient Descent",
         "Derivation of the normal equation, batch vs stochastic gradient descent, learning rate schedules, and mean squared error loss.",
         ["interactive", "video"], ["beginner", "intermediate"]),
        ("Logistic Regression and Classification Decision Boundaries",
         "Sigmoid activation, binary cross-entropy loss, odds ratios, decision boundaries, and multi-class One-vs-Rest classification.",
         ["reading", "practice"], ["beginner", "intermediate"]),
        ("Regularization Techniques: Ridge (L2), Lasso (L1), and Elastic Net",
         "Preventing overfitting in high-dimensional models. Geometric intuition of L1 sparsity versus L2 weight shrinkage.",
         ["video", "interactive"], ["intermediate"]),
        ("The Bias-Variance Tradeoff and Model Generalization",
         "Decomposing expected prediction error into bias, variance, and irreducible noise. Overfitting vs underfitting diagnostic curves.",
         ["reading", "video"], ["intermediate"]),
        ("Decision Trees, Information Gain, and Gini Impurity",
         "Recursive binary splitting, entropy calculations, CART algorithm, tree pruning strategies, and handling continuous features.",
         ["interactive", "practice"], ["beginner", "intermediate"]),
        ("Ensemble Learning: Random Forests and Bagging",
         "Bootstrap aggregating, feature bagging for decorrelation, out-of-bag error estimation, and feature importance rankings.",
         ["video", "reading"], ["intermediate"]),
        ("Gradient Boosted Decision Trees (GBDT, XGBoost, LightGBM)",
         "Boosting theory: fitting successive trees to negative gradients of pseudo-residuals. Learning rate shrinkage and tree depth tuning.",
         ["reading", "interactive"], ["intermediate", "advanced"]),
        ("Support Vector Machines (SVM) and Kernel Trick",
         "Maximum margin hyperplanes, soft margin slack variables, dual Lagrangian formulation, and radial basis function (RBF) kernels.",
         ["video", "practice"], ["intermediate", "advanced"]),
        ("Unsupervised Learning: K-Means Clustering and Hierarchical Clustering",
         "Lloyd's algorithm, Voronoi cells, inertia elbow curves, silhouette analysis, and agglomerative clustering with dendrograms.",
         ["interactive", "practice"], ["beginner", "intermediate"]),
        ("Model Evaluation Metrics: ROC-AUC, Precision-Recall, and F1-Score",
         "Beyond accuracy: confusion matrices, precision, recall, specificity, F-beta scores, macro/micro averages, and PR curves under class imbalance.",
         ["practice", "reading"], ["intermediate"]),
    ],
    "deep_learning": [
        ("Perceptrons to Multi-Layer Perceptrons (MLPs)",
         "Linear separability limitations (XOR problem), hidden layers, forward propagation, and universal approximation theorems.",
         ["video", "reading"], ["beginner", "intermediate"]),
        ("Activation Functions: Sigmoid, Tanh, ReLU, and GELU",
         "Vanishing and exploding gradient issues, dead neuron syndrome in ReLU, smooth non-linearities, and empirical benchmark comparisons.",
         ["reading", "interactive"], ["intermediate"]),
        ("Backpropagation and Computational Graphs Demystified",
         "Step-by-step multivariate chain rule on computation graphs. Deriving weight gradient updates for deep neural architectures.",
         ["interactive", "video"], ["intermediate", "advanced"]),
        ("Deep Learning Optimization: Adam, RMSprop, and Momentum",
         "Escaping saddle points, exponentially weighted moving averages of gradients, adaptive learning rates, and weight decay vs L2.",
         ["reading", "practice"], ["intermediate", "advanced"]),
        ("Convolutional Neural Networks (CNNs) for Computer Vision",
         "Convolution kernels, strides, padding, feature map dimensions, max pooling, receptive fields, and residual connections (ResNet).",
         ["video", "interactive"], ["intermediate"]),
        ("Batch Normalization, Layer Normalization, and Dropout",
         "Internal covariate shift alleviation, regularization via stochastic neuron deactivation, and normalization across sequence dimensions.",
         ["reading", "video"], ["intermediate"]),
        ("Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM)",
         "Handling sequential data, hidden states, vanishing gradients in backpropagation through time, input/forget/output gates in LSTMs.",
         ["video", "practice"], ["intermediate", "advanced"]),
        ("The Transformer Architecture: Self-Attention Mechanics",
         "Scaled dot-product attention, queries, keys, values, multi-head attention projection, positional encodings, and parallel training.",
         ["interactive", "video"], ["advanced"]),
        ("Transfer Learning and Fine-Tuning Pretrained Vision and Language Models",
         "Feature extraction vs full fine-tuning, freezing layer weights, learning rate warmups, and LoRA parameter-efficient adaptation.",
         ["practice", "reading"], ["intermediate", "advanced"]),
        ("PyTorch Essentials: Tensors, Autograd, and Model Training Loops",
         "Hands-on PyTorch: torch.Tensor operations, autograd computational history, nn.Module subclassing, dataset dataloaders, and training loops.",
         ["practice", "interactive"], ["beginner", "intermediate"]),
    ],
    "databases_sql": [
        ("Relational Database Design and Entity-Relationship Modeling",
         "Entities, attributes, relationships (1:1, 1:N, N:M), primary keys, foreign key constraints, and relational schema diagrams.",
         ["reading", "video"], ["beginner"]),
        ("SQL Fundamentals: SELECT, WHERE, ORDER BY, and LIMIT",
         "Writing declarative queries to filter rows, sort outputs, handle NULL values, and format result sets effectively.",
         ["practice", "interactive"], ["beginner"]),
        ("Mastering SQL Joins: INNER, LEFT, RIGHT, and FULL OUTER",
         "Combining tables across relational keys. Venn diagram intuition versus Cartesian cross-product filtration mechanics.",
         ["interactive", "practice"], ["beginner", "intermediate"]),
        ("SQL Aggregations: GROUP BY, HAVING, and Aggregate Functions",
         "COUNT, SUM, AVG, MIN, MAX calculations. Filtering aggregated groups with HAVING versus row-level filtering with WHERE.",
         ["practice", "reading"], ["intermediate"]),
        ("Subqueries and Common Table Expressions (CTEs)",
         "Correlated vs non-correlated subqueries, readability improvements using WITH clauses, and recursive CTEs for hierarchical tree queries.",
         ["video", "practice"], ["intermediate"]),
        ("SQL Window Functions: RANK, DENSE_RANK, ROW_NUMBER, and LAG/LEAD",
         "Performing calculations across rows related to the current row without collapsing groups. OVER (PARTITION BY ... ORDER BY ...).",
         ["interactive", "practice"], ["intermediate", "advanced"]),
        ("Database Normalization: 1NF, 2NF, 3NF, and BCNF",
         "Eliminating data redundancy and update/delete anomalies. Functional dependencies, lossless decomposition, and dependency preservation.",
         ["reading", "video"], ["intermediate"]),
        ("Database Indexing: B-Trees, Hash Indexes, and Query Plans",
         "How B-Tree indexes speed up lookups and range scans. Composite index column ordering, index selectivity, and EXPLAIN query analysis.",
         ["video", "reading"], ["intermediate", "advanced"]),
        ("Transactions and ACID Guarantees: Atomicity to Durability",
         "Transaction boundaries (BEGIN, COMMIT, ROLLBACK), write-ahead logging (WAL), dirty reads, non-repeatable reads, and isolation levels.",
         ["reading", "video"], ["intermediate", "advanced"]),
        ("NoSQL vs SQL: Document, Key-Value, and Columnar Stores",
         "CAP theorem, horizontal partitioning (sharding), eventual consistency, and choosing between PostgreSQL, MongoDB, and Redis.",
         ["reading", "interactive"], ["intermediate"]),
    ],
    "web_development": [
        ("How the Web Works: DNS, TCP/IP, TLS, and HTTP/1.1 to HTTP/3",
         "The lifecycle of a web request. Domain resolution, SSL/TLS handshakes, HTTP methods, headers, status codes, and multiplexing.",
         ["video", "reading"], ["beginner", "intermediate"]),
        ("RESTful API Design Best Practices and Status Codes",
         "Resource-oriented URIs, HTTP verbs (GET, POST, PUT, PATCH, DELETE), idempotency, semantic status codes, and API versioning.",
         ["reading", "practice"], ["beginner", "intermediate"]),
        ("Building High-Performance APIs with FastAPI and Pydantic",
         "Type hints, automatic schema validation, dependency injection, async request handling, and interactive Swagger documentation.",
         ["interactive", "practice"], ["intermediate"]),
        ("Authentication and Authorization: JWT, Cookies, and OAuth 2.0",
         "Stateless JSON Web Tokens, HMAC and RSA signatures, HTTP-only secure cookies, refresh tokens, and third-party OAuth flows.",
         ["video", "practice"], ["intermediate", "advanced"]),
        ("Asynchronous Programming and Background Workers in Python",
         "Non-blocking I/O event loops, Celery task queues, Redis message brokers, scheduling periodic jobs, and handling rate limits.",
         ["practice", "reading"], ["intermediate"]),
        ("Cross-Origin Resource Sharing (CORS) and Web Security Headers",
         "Same-Origin Policy, preflight OPTIONS requests, Access-Control-Allow headers, Content Security Policy (CSP), and CSRF protection.",
         ["reading", "video"], ["intermediate"]),
        ("WebSockets and Server-Sent Events (SSE) for Real-Time Apps",
         "Full-duplex bidirectional communication with WebSockets versus unidirectional server streams for real-time live updates.",
         ["interactive", "practice"], ["intermediate", "advanced"]),
        ("API Rate Limiting, Throttling, and Denial of Service Defense",
         "Token bucket and leaky bucket algorithms, Redis atomic counters, sliding window rate limiters, and protecting upstream resources.",
         ["reading", "video"], ["intermediate", "advanced"]),
        ("Microservices Architecture and Inter-Service Communication",
         "Decoupling monoliths, service discovery, API gateways, synchronous REST/gRPC versus asynchronous event-driven messaging.",
         ["video", "reading"], ["advanced"]),
        ("Docker Containerization for Web Applications and Services",
         "Writing efficient Dockerfiles, multi-stage builds, container networking, volume mounts, and orchestrating multi-container stacks.",
         ["practice", "interactive"], ["intermediate"]),
    ],
    "system_design": [
        ("Horizontal vs Vertical Scaling and Load Balancing Strategies",
         "Scaling out versus scaling up. Layer 4 vs Layer 7 load balancers, round-robin, least connections, and consistent hashing algorithms.",
         ["video", "reading"], ["intermediate"]),
        ("Caching Patterns and Cache Invalidation (Cache-Aside, Write-Through)",
         "Reducing database load with Redis/Memcached. Eviction policies (LRU, LFU), thundering herd problem, and cache stampede solutions.",
         ["reading", "practice"], ["intermediate", "advanced"]),
        ("Database Sharding, Partitioning, and Replication",
         "Master-replica read scaling, master-master replication, range-based vs hash-based sharding, cross-shard joins, and distributed transactions.",
         ["video", "interactive"], ["advanced"]),
        ("Message Queues and Event-Driven Architecture (Kafka & RabbitMQ)",
         "Decoupling producers and consumers, pub-sub semantics, message ordering, consumer groups, idempotency, and dead-letter queues.",
         ["video", "reading"], ["intermediate", "advanced"]),
        ("Consistent Hashing and Distributed Hash Tables",
         "Partitioning keys across dynamic nodes with minimal remapping upon node failures or additions. Virtual nodes and replication rings.",
         ["interactive", "reading"], ["advanced"]),
        ("CAP Theorem and Distributed Data Consistency Models",
         "Consistency, Availability, and Partition Tolerance tradeoffs. Strong consistency, sequential consistency, and eventual consistency.",
         ["reading", "video"], ["intermediate", "advanced"]),
        ("Rate Limiting, Circuit Breakers, and Fault Tolerance in Systems",
         "Preventing cascading system outages. Hystrix pattern, circuit states (closed, open, half-open), fallback strategies, and graceful degradation.",
         ["reading", "practice"], ["intermediate", "advanced"]),
        ("Content Delivery Networks (CDNs) and Static Asset Optimization",
         "Edge points of presence, geographic DNS routing, origin shield caching, HTTP caching headers (Cache-Control, ETag), and latency cuts.",
         ["video", "reading"], ["beginner", "intermediate"]),
        ("Designing a Scalable URL Shortener (TinyURL Deep Dive)",
         "Step-by-step system design interview blueprint: capacity estimation, Base62 encoding, unique ID generation (Snowflake), and DB schema.",
         ["interactive", "practice"], ["intermediate"]),
        ("Designing a Real-Time Collaborative Notification System",
         "High-throughput fan-out architectures, user notification preferences, rate limits, WebSocket push connections, and storage schemas.",
         ["reading", "practice"], ["advanced"]),
    ],
}


def generate_resources(total_count: int = 300) -> List[LearningResource]:
    """
    Generates a deterministic catalog of learning resources with realistic metadata.
    Guarantees exactly `total_count` resources across the curriculum topics.
    """
    rng = random.Random(42)
    resources: List[LearningResource] = []

    # First, unpack base templates (100 distinct rich resources)
    all_blueprints = []
    for topic_id, templates in _TOPIC_TEMPLATES.items():
        for idx, (title, desc, formats, diffs) in enumerate(templates):
            all_blueprints.append((topic_id, title, desc, formats, diffs, idx))

    # Sources pool
    sources = [
        "MIT OpenCourseWare", "Stanford Online", "Harvard CS", "DeepLearning.AI",
        "Carnegie Mellon University", "UC Berkeley EECS", "Google Developers",
        "Coursera Academic Hub", "edX Institute", "Kaggle Learn"
    ]

    res_counter = 1
    # Expand to 300 resources (each base template has 3 variations across formats/difficulties/focus)
    variations = [
        ("Comprehensive Guide: ", " Deep conceptual walk-through with real-world examples and code snippets."),
        ("Hands-on Practice Lab: ", " Interactive exercises and challenge sets designed to reinforce foundational mastery."),
        ("Advanced Masterclass: ", " Rigorous deep dive covering edge cases, performance profiling, and systems trade-offs.")
    ]

    for v_idx, (prefix, suffix) in enumerate(variations):
        for topic_id, base_title, base_desc, formats, diffs, b_idx in all_blueprints:
            if len(resources) >= total_count:
                break

            # Alternate secondary topic tag for some resources to enable cross-domain hybrid recommendations
            topics = [topic_id]
            if (res_counter % 3 == 0):
                # Pick a related secondary topic
                related_map = {
                    "python_basics": "data_structures",
                    "data_structures": "algorithms",
                    "algorithms": "machine_learning",
                    "linear_algebra": "machine_learning",
                    "probability_stats": "machine_learning",
                    "machine_learning": "deep_learning",
                    "deep_learning": "python_basics",
                    "databases_sql": "web_development",
                    "web_development": "system_design",
                    "system_design": "databases_sql"
                }
                sec = related_map.get(topic_id)
                if sec and sec not in topics:
                    topics.append(sec)

            fmt = formats[v_idx % len(formats)]
            diff = diffs[v_idx % len(diffs)]
            est_mins = rng.choice([15, 20, 25, 30, 40, 45, 60])
            source = sources[(res_counter + v_idx) % len(sources)]

            title = f"{prefix}{base_title}" if v_idx > 0 else base_title
            desc = f"{base_desc}{suffix}" if v_idx > 0 else base_desc

            res_id = f"res_{res_counter:03d}"
            resources.append(
                LearningResource(
                    resource_id=res_id,
                    title=title,
                    description=desc,
                    topics=topics,
                    format=fmt,
                    difficulty=diff,
                    estimated_minutes=est_mins,
                    source=source
                )
            )
            res_counter += 1

    return resources[:total_count]


# Instantiated singleton catalog
CATALOG: List[LearningResource] = generate_resources(300)
CATALOG_BY_ID: Dict[str, LearningResource] = {r.resource_id: r for r in CATALOG}
