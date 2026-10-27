"""
Curated diagnostic and practice question bank for Bayesian Knowledge Tracing.
Covers all 10 topics with multiple-choice questions, answer keys, and pedagogical explanations.
"""

from dataclasses import dataclass
from typing import List, Dict


@dataclass
class QuizQuestion:
    question_id: str
    topic_id: str
    text: str
    options: List[str]
    correct_index: int
    explanation: str
    difficulty: str = "intermediate"

    def is_correct(self, selected_index: int) -> bool:
        return selected_index == self.correct_index


QUESTION_BANK: List[QuizQuestion] = [
    # 1. Python Fundamentals
    QuizQuestion(
        question_id="q_py_01",
        topic_id="python_basics",
        text="What is the output of `bool([])`, `bool([0])`, and `bool('')` in Python?",
        options=["(False, False, False)", "(False, True, False)", "(True, True, True)", "(False, True, True)"],
        correct_index=1,
        explanation="Empty containers like `[]` and empty strings `''` evaluate to False (falsy). A non-empty list `[0]` evaluates to True (truthy) regardless of its elements.",
        difficulty="beginner",
    ),
    QuizQuestion(
        question_id="q_py_02",
        topic_id="python_basics",
        text="Which keyword is used inside a nested function to rebind a variable in the nearest enclosing non-global scope?",
        options=["global", "nonlocal", "outer", "super"],
        correct_index=1,
        explanation="The `nonlocal` keyword allows re-assigning identifiers defined in an outer, enclosing lexical scope that is not global.",
        difficulty="intermediate",
    ),
    QuizQuestion(
        question_id="q_py_03",
        topic_id="python_basics",
        text="What happens if a default argument in a Python function definition is a mutable object like a list `def fn(x=[])`?",
        options=[
            "A new list is instantiated on every function invocation.",
            "The list is created once at function definition time and mutated across subsequent calls.",
            "Python raises a `MutableDefaultArgumentError` during parsing.",
            "The argument defaults to `None` automatically."
        ],
        correct_index=1,
        explanation="Default parameter values are evaluated once when the function definition is executed, leading to shared state across invocations.",
        difficulty="intermediate",
    ),
    QuizQuestion(
        question_id="q_py_04",
        topic_id="python_basics",
        text="What is the primary architectural purpose of the Global Interpreter Lock (GIL) in standard CPython?",
        options=[
            "To prevent deadlocks by serializing memory allocation and reference counting across native OS threads.",
            "To speed up execution of pure numerical calculations across all available CPU cores.",
            "To sandbox external C extensions from corrupting host operating system files.",
            "To automate garbage collection without requiring cyclic reference sweeps."
        ],
        correct_index=0,
        explanation="The CPython GIL synchronizes thread execution so that CPython's reference counting memory management remains thread-safe without fine-grained locks.",
        difficulty="advanced",
    ),

    # 2. Data Structures
    QuizQuestion(
        question_id="q_ds_01",
        topic_id="data_structures",
        text="What is the worst-case time complexity of searching for an element in an unbalanced Binary Search Tree of N nodes?",
        options=["O(1)", "O(log N)", "O(N)", "O(N log N)"],
        correct_index=2,
        explanation="When elements are inserted in sorted order, an unbalanced BST degenerates into a linear linked list with O(N) search complexity.",
        difficulty="beginner",
    ),
    QuizQuestion(
        question_id="q_ds_02",
        topic_id="data_structures",
        text="How is collision handled in open addressing with linear probing?",
        options=[
            "By storing collided keys in a separate linked list attached to the bucket.",
            "By probing subsequent contiguous slots `(hash + i) % capacity` until an empty slot is encountered.",
            "By rehashing the table and doubling the number of prime buckets immediately.",
            "By dropping the older key and overwriting it with the newer element."
        ],
        correct_index=1,
        explanation="Linear probing checks consecutive array slots sequentially until an unoccupied cell is discovered.",
        difficulty="intermediate",
    ),
    QuizQuestion(
        question_id="q_ds_03",
        topic_id="data_structures",
        text="In a binary min-heap stored in an array with 0-based indexing, what are the indices of the children of node at index `i`?",
        options=["2i and 2i + 1", "2i + 1 and 2i + 2", "i/2 and i/2 + 1", "2i - 1 and 2i"],
        correct_index=1,
        explanation="In 0-based heap indexing, the left child is at `2i + 1` and the right child is at `2i + 2`. The parent is at `(i - 1) // 2`.",
        difficulty="intermediate",
    ),
    QuizQuestion(
        question_id="q_ds_04",
        topic_id="data_structures",
        text="What is the amortized time complexity per operation in a Disjoint Set Union (Union-Find) structure with both path compression and union by rank?",
        options=["O(1)", "O(α(N)) (Inverse Ackermann function)", "O(log N)", "O(√N)"],
        correct_index=1,
        explanation="Path compression combined with union by rank guarantees nearly constant O(α(N)) amortized time, where α(N) ≤ 4 for all practical universe sizes.",
        difficulty="advanced",
    ),

    # 3. Algorithms & Complexity
    QuizQuestion(
        question_id="q_algo_01",
        topic_id="algorithms",
        text="What is the recurrence relation for standard MergeSort, and what asymptotic bound does Master Theorem yield?",
        options=[
            "T(N) = T(N-1) + O(N) -> O(N²)",
            "T(N) = 2T(N/2) + O(N) -> O(N log N)",
            "T(N) = 2T(N/2) + O(1) -> O(N)",
            "T(N) = 4T(N/2) + O(N²) -> O(N² log N)"
        ],
        correct_index=1,
        explanation="MergeSort divides an array into two halves of size N/2 and merges them in O(N) time: T(N) = 2T(N/2) + O(N), yielding O(N log N).",
        difficulty="beginner",
    ),
    QuizQuestion(
        question_id="q_algo_02",
        topic_id="algorithms",
        text="Why does Dijkstra's algorithm fail to find the shortest path on graphs containing negative edge weights?",
        options=[
            "It gets trapped in infinite cycles even if there is no negative cycle.",
            "Once a vertex's shortest distance is extracted from the priority queue, Dijkstra assumes its distance is finalized and never relaxes it again.",
            "Dijkstra's algorithm can only operate on undirected unweighted graphs.",
            "The priority queue raises a domain error when handling negative keys."
        ],
        correct_index=1,
        explanation="Dijkstra's greedy assumption posits that paths only grow longer; negative weights violate this property, requiring Bellman-Ford instead.",
        difficulty="intermediate",
    ),
    QuizQuestion(
        question_id="q_algo_03",
        topic_id="algorithms",
        text="In dynamic programming, what two core properties must a problem exhibit to be solvable via DP?",
        options=[
            "Greedy choice property and divide-and-conquer symmetry.",
            "Optimal substructure and overlapping subproblems.",
            "Monotonic convergence and non-overlapping subproblems.",
            "Polynomial verifiability and NP-completeness."
        ],
        correct_index=1,
        explanation="Optimal substructure means an optimal solution contains optimal sub-solutions; overlapping subproblems means the same sub-instances recur repeatedly.",
        difficulty="intermediate",
    ),
    QuizQuestion(
        question_id="q_algo_04",
        topic_id="algorithms",
        text="What is the worst-case time complexity of Knuth-Morris-Pratt (KMP) string matching for text length N and pattern length M?",
        options=["O(N × M)", "O(N + M)", "O(N log M)", "O(M²)"],
        correct_index=1,
        explanation="KMP computes a prefix function pi-table in O(M) time and scans the text without backtracking in O(N) time, giving O(N + M).",
        difficulty="advanced",
    ),

    # 4. Linear Algebra
    QuizQuestion(
        question_id="q_la_01",
        topic_id="linear_algebra",
        text="What does a determinant of zero indicate about a square matrix A?",
        options=[
            "The matrix is symmetric and positive definite.",
            "The columns are linearly dependent and the matrix is non-invertible (singular).",
            "All eigenvalues of matrix A are strictly equal to 1.",
            "The matrix represents an orthogonal rotation in Euclidean space."
        ],
        correct_index=1,
        explanation="det(A) = 0 indicates that the matrix squashes space into a lower dimension, its columns are linearly dependent, and no unique inverse exists.",
        difficulty="beginner",
    ),
    QuizQuestion(
        question_id="q_la_02",
        topic_id="linear_algebra",
        text="If vector v is an eigenvector of matrix A with eigenvalue λ, which equation holds?",
        options=["Av = λv", "Aλ = v", "A + v = λI", "vᵀAv = λ"],
        correct_index=0,
        explanation="By definition, multiplying an eigenvector v by transformation matrix A merely scales it by a scalar factor λ: Av = λv.",
        difficulty="beginner",
    ),
    QuizQuestion(
        question_id="q_la_03",
        topic_id="linear_algebra",
        text="In the Singular Value Decomposition (SVD) of an m×n matrix A = U Σ Vᵀ, what do the columns of U represent?",
        options=[
            "The eigenvectors of VᵀV.",
            "The orthonormal eigenvectors of AAᵀ (left singular vectors).",
            "The diagonal eigenvalues of matrix A.",
            "The null space basis of matrix A."
        ],
        correct_index=1,
        explanation="In SVD, U contains the orthonormal eigenvectors of the m×m matrix AAᵀ, known as the left singular vectors.",
        difficulty="intermediate",
    ),
    QuizQuestion(
        question_id="q_la_04",
        topic_id="linear_algebra",
        text="A real symmetric matrix A is positive semi-definite if and only if:",
        options=[
            "All of its off-diagonal entries are strictly non-negative.",
            "All of its eigenvalues are greater than or equal to zero (λᵢ ≥ 0).",
            "Its determinant is strictly negative.",
            "The trace of A equals its rank."
        ],
        correct_index=1,
        explanation="For a symmetric matrix, positive semi-definiteness is equivalent to xᵀAx ≥ 0 for all x, which occurs if and only if all eigenvalues λ ≥ 0.",
        difficulty="advanced",
    ),

    # 5. Probability & Statistics
    QuizQuestion(
        question_id="q_prob_01",
        topic_id="probability_stats",
        text="According to Bayes' Theorem, how is posterior probability P(A|B) calculated?",
        options=[
            "P(A|B) = [P(B|A) × P(A)] / P(B)",
            "P(A|B) = [P(A) × P(B)] / P(B|A)",
            "P(A|B) = P(A ∩ B) × P(B)",
            "P(A|B) = P(A) + P(B) - P(A ∩ B)"
        ],
        correct_index=0,
        explanation="Bayes' formula states: P(A|B) = P(B|A)P(A) / P(B), where P(A) is the prior, P(B|A) is likelihood, and P(B) is marginal evidence.",
        difficulty="beginner",
    ),
    QuizQuestion(
        question_id="q_prob_02",
        topic_id="probability_stats",
        text="What does the Central Limit Theorem (CLT) state regarding the distribution of sample means?",
        options=[
            "The population distribution must be normal for sample means to be normal.",
            "As sample size n increases, the sample mean distribution approaches normality regardless of the original population shape (given finite variance).",
            "The sample variance approaches zero when n exceeds 30.",
            "Individual observations in any large sample automatically become normally distributed."
        ],
        correct_index=1,
        explanation="CLT proves that the sum or average of independent random variables with finite variance asymptotically follows a Gaussian distribution.",
        difficulty="intermediate",
    ),
    QuizQuestion(
        question_id="q_prob_03",
        topic_id="probability_stats",
        text="What is a Type I error in statistical hypothesis testing?",
        options=[
            "Rejecting the null hypothesis H₀ when H₀ is actually true (false positive).",
            "Failing to reject H₀ when the alternative H₁ is actually true (false negative).",
            "Selecting a sample that is smaller than the required statistical power threshold.",
            "Violating homoscedasticity in linear regression residuals."
        ],
        correct_index=0,
        explanation="Type I error occurs when a true null hypothesis is incorrectly rejected, with probability controlled by significance level α.",
        difficulty="intermediate",
    ),
    QuizQuestion(
        question_id="q_prob_04",
        topic_id="probability_stats",
        text="Which distribution serves as the conjugate prior for the probability parameter p of a Binomial likelihood?",
        options=["Gaussian distribution", "Beta distribution", "Poisson distribution", "Gamma distribution"],
        correct_index=1,
        explanation="The Beta distribution is conjugate to the Binomial likelihood because multiplying them results in an updated posterior that is also a Beta distribution.",
        difficulty="advanced",
    ),

    # 6. Machine Learning Foundations
    QuizQuestion(
        question_id="q_ml_01",
        topic_id="machine_learning",
        text="In Ridge (L2) vs Lasso (L1) regularization, which one induces exact zero coefficients for feature selection?",
        options=[
            "Ridge regression because the spherical constraint tangent touches along the axes.",
            "Lasso regression because its diamond-shaped L1 norm constraint has sharp corners on the coordinate axes.",
            "Both induce identical sparsity patterns under normalized inputs.",
            "Neither, as only Principal Component Analysis performs feature selection."
        ],
        correct_index=1,
        explanation="Lasso's L1 diamond geometry intersects residual sum of squares contours directly on coordinate axes, zeroing out non-informative coefficients.",
        difficulty="intermediate",
    ),
    QuizQuestion(
        question_id="q_ml_02",
        topic_id="machine_learning",
        text="How does high model complexity typically affect bias and variance?",
        options=[
            "High bias and high variance",
            "Low bias and high variance (overfitting risk)",
            "High bias and low variance (underfitting risk)",
            "Low bias and low variance"
        ],
        correct_index=1,
        explanation="Complex models fit training data closely (low bias) but are sensitive to sampling noise in unseen test data (high variance).",
        difficulty="beginner",
    ),
    QuizQuestion(
        question_id="q_ml_03",
        topic_id="machine_learning",
        text="What is the difference between bagging and boosting ensembles?",
        options=[
            "Bagging trains models sequentially on residuals; boosting trains independent models in parallel on bootstrapped samples.",
            "Bagging trains independent models in parallel to reduce variance; boosting trains models sequentially to reduce bias by focusing on previous mistakes.",
            "Bagging only works on neural networks, while boosting only works on linear models.",
            "Bagging maximizes entropy; boosting minimizes covariance."
        ],
        correct_index=1,
        explanation="Bagging (e.g. Random Forest) averages independent predictors to curb variance, whereas Boosting (e.g. XGBoost) builds sequential correctors to reduce bias.",
        difficulty="intermediate",
    ),
    QuizQuestion(
        question_id="q_ml_04",
        topic_id="machine_learning",
        text="Under extreme class imbalance (e.g., 99.9% negative), which evaluation metric provides the most reliable performance view?",
        options=["Accuracy", "Precision-Recall AUC (PR-AUC)", "Specificity alone", "Mean Squared Error"],
        correct_index=1,
        explanation="Accuracy and ROC-AUC can be deceptively optimistic when the true negative class overwhelms the dataset; PR-AUC focuses on the positive minority class.",
        difficulty="advanced",
    ),

    # 7. Deep Learning & Neural Networks
    QuizQuestion(
        question_id="q_dl_01",
        topic_id="deep_learning",
        text="Why did Rectified Linear Units (ReLU) largely supersede Sigmoid activations in hidden layers of deep networks?",
        options=[
            "ReLU is bounded between -1 and +1, guaranteeing numerical stability.",
            "ReLU maintains a constant derivative of 1 for positive activations, mitigating the vanishing gradient problem during backpropagation.",
            "ReLU is completely differentiable at zero.",
            "ReLU prevents all dead neurons from occurring."
        ],
        correct_index=1,
        explanation="Sigmoids saturate near 0 and 1 with near-zero derivatives, causing gradients to vanish across deep layers; ReLU maintains gradient 1 for positive inputs.",
        difficulty="beginner",
    ),
    QuizQuestion(
        question_id="q_dl_02",
        topic_id="deep_learning",
        text="In the Adam optimizer, what two moving averages are tracked for each parameter?",
        options=[
            "The maximum and minimum gradients.",
            "The exponentially decaying average of past gradients (first moment) and past squared gradients (second moment).",
            "The training loss and validation accuracy slopes.",
            "The Hessian diagonal elements and momentum pivots."
        ],
        correct_index=1,
        explanation="Adam combines Momentum (first raw moment) and RMSprop (second uncentered moment) with bias correction for adaptive per-parameter learning rates.",
        difficulty="intermediate",
    ),
    QuizQuestion(
        question_id="q_dl_03",
        topic_id="deep_learning",
        text="In the standard Transformer self-attention formula `Attention(Q, K, V) = softmax(QKᵀ / √d_k) V`, what is the purpose of dividing by `√d_k`?",
        options=[
            "To normalize the matrix rank to 1.",
            "To prevent the dot products from growing excessively large in high dimensions, which would push softmax into regions with vanishing gradients.",
            "To enforce orthogonality between query and key projections.",
            "To calculate the Euclidean cosine distance between tokens."
        ],
        correct_index=1,
        explanation="For large d_k, dot products grow in magnitude, driving softmax into regions with tiny gradients; scaling by 1/√d_k stabilizes gradient flow.",
        difficulty="advanced",
    ),
    QuizQuestion(
        question_id="q_dl_04",
        topic_id="deep_learning",
        text="What design mechanism enabled Deep Residual Networks (ResNet) to train networks with hundreds of layers?",
        options=[
            "Replacing all convolutions with dense fully connected projections.",
            "Identity skip connections that allow gradients to propagate directly through the residual path F(x) + x.",
            "Eliminating nonlinear activation functions.",
            "Using purely deterministic stochastic weight averaging."
        ],
        correct_index=1,
        explanation="Skip connections allow gradients to flow unimpeded directly back through the identity shortcut, overcoming degradation in very deep networks.",
        difficulty="intermediate",
    ),

    # 8. Relational Databases & SQL
    QuizQuestion(
        question_id="q_sql_01",
        topic_id="databases_sql",
        text="What is the key difference between the `WHERE` clause and the `HAVING` clause in SQL?",
        options=[
            "`WHERE` filters aggregated groups; `HAVING` filters individual rows prior to grouping.",
            "`WHERE` filters individual records before grouping; `HAVING` filters grouped rows after aggregation functions have executed.",
            "`WHERE` can only be used with primary keys; `HAVING` works with foreign keys.",
            "`WHERE` requires an index; `HAVING` operates on tables in memory."
        ],
        correct_index=1,
        explanation="WHERE evaluates condition expressions row-by-row before GROUP BY; HAVING applies filter predicates to grouped summary rows.",
        difficulty="beginner",
    ),
    QuizQuestion(
        question_id="q_sql_02",
        topic_id="databases_sql",
        text="Which SQL window function computes a rank where ties receive the same rank and subsequent ranks are non-consecutive (with gaps)?",
        options=["ROW_NUMBER()", "DENSE_RANK()", "RANK()", "NTILE()"],
        correct_index=2,
        explanation="RANK() leaves gaps after ties (e.g., 1, 2, 2, 4), whereas DENSE_RANK() does not leave gaps (1, 2, 2, 3), and ROW_NUMBER() assigns unique sequential integers.",
        difficulty="intermediate",
    ),
    QuizQuestion(
        question_id="q_sql_03",
        topic_id="databases_sql",
        text="A relational schema is in Third Normal Form (3NF) if it is in 2NF and:",
        options=[
            "Every column allows NULL values.",
            "There are no transitive functional dependencies of non-prime attributes on candidate keys.",
            "All relationships are strictly one-to-one.",
            "Composite primary keys are banned."
        ],
        correct_index=1,
        explanation="3NF eliminates transitive dependencies: non-key attributes must depend solely on the candidate key and nothing else.",
        difficulty="intermediate",
    ),
    QuizQuestion(
        question_id="q_sql_04",
        topic_id="databases_sql",
        text="In database transaction management, what does the 'I' in ACID stand for, and what anomaly does Serializable isolation prevent?",
        options=[
            "Idempotency; prevents duplicate inserts.",
            "Isolation; prevents phantom reads, non-repeatable reads, and dirty reads.",
            "Integrity; prevents foreign key deletions.",
            "Indexing; prevents full table scans."
        ],
        correct_index=1,
        explanation="Isolation guarantees concurrent transactions execute as if serially ordered, preventing dirty reads, unrepeatable reads, and phantoms.",
        difficulty="advanced",
    ),

    # 9. Web Architecture & APIs
    QuizQuestion(
        question_id="q_web_01",
        topic_id="web_development",
        text="Which HTTP methods are defined as idempotent according to the HTTP/1.1 specification?",
        options=[
            "POST and PATCH",
            "GET, PUT, DELETE, and HEAD",
            "Only POST",
            "Only GET"
        ],
        correct_index=1,
        explanation="An idempotent method produces the same server state regardless of whether executed once or repeatedly: GET, PUT, DELETE, and HEAD are idempotent; POST is not.",
        difficulty="beginner",
    ),
    QuizQuestion(
        question_id="q_web_02",
        topic_id="web_development",
        text="What browser mechanism triggers an HTTP preflight `OPTIONS` request before sending the actual request?",
        options=[
            "Same-Origin Policy violation without credentials.",
            "Cross-Origin Resource Sharing (CORS) when making a request with custom headers or non-simple HTTP methods (e.g., PUT, DELETE).",
            "DNS caching expiration in the browser networking stack.",
            "A missing Content-Type header on a standard GET request."
        ],
        correct_index=1,
        explanation="Browsers send an OPTIONS preflight to verify that the target cross-origin server permits the requested method and custom headers.",
        difficulty="intermediate",
    ),
    QuizQuestion(
        question_id="q_web_03",
        topic_id="web_development",
        text="Where should sensitive session tokens (like JWTs) be stored in a web client to mitigate Cross-Site Scripting (XSS) extraction?",
        options=[
            "In `localStorage` or `sessionStorage` accessible by JavaScript.",
            "In an `HttpOnly`, `Secure`, `SameSite=Strict` cookie.",
            "In the URL query string parameters.",
            "In a global JavaScript `window` object variable."
        ],
        correct_index=1,
        explanation="HttpOnly cookies cannot be read by clientside JavaScript document.cookie APIs, shielding the session token from XSS script exfiltration.",
        difficulty="intermediate",
    ),
    QuizQuestion(
        question_id="q_web_04",
        topic_id="web_development",
        text="What is the fundamental architectural difference between WebSockets and Server-Sent Events (SSE)?",
        options=[
            "WebSockets provide bidirectional, full-duplex TCP communication; SSE provides unidirectional server-to-client streaming over standard HTTP.",
            "SSE requires custom binary protocol parsers; WebSockets rely purely on XML.",
            "WebSockets can only handle plaintext; SSE natively transmits raw binary frames.",
            "There is no difference; they are synonymous standards."
        ],
        correct_index=0,
        explanation="WebSockets upgrade the connection to a bidirectional TCP channel; SSE maintains an open HTTP connection for unidirectional push from server to browser.",
        difficulty="advanced",
    ),

    # 10. System Design & Distributed Systems
    QuizQuestion(
        question_id="q_sys_01",
        topic_id="system_design",
        text="In the CAP theorem for distributed systems, what does the theorem state when a network partition (P) occurs?",
        options=[
            "A distributed system can guarantee both Consistency (C) and Availability (A).",
            "The system must choose between either Consistency (C) or Availability (A).",
            "The system must drop all partitions immediately.",
            "Linearizability is guaranteed at all times."
        ],
        correct_index=1,
        explanation="When network partitions occur, a distributed data store must either return errors/time out (preserving C) or return potentially stale local data (preserving A).",
        difficulty="beginner",
    ),
    QuizQuestion(
        question_id="q_sys_02",
        topic_id="system_design",
        text="How does consistent hashing minimize data migration when a cache node is added or removed from a cluster?",
        options=[
            "By broadcasting all keys to every single machine simultaneously.",
            "By mapping both server nodes and keys onto a circular hash ring, so adding a node only affects adjacent keys rather than all N keys.",
            "By computing `hash(key) % N` where N is the updated server count.",
            "By writing all keys to a persistent centralized master relational database."
        ],
        correct_index=1,
        explanation="Consistent hashing maps keys and nodes to a ring; on node addition or removal, only keys in the immediate neighborhood must move (average K/N keys).",
        difficulty="intermediate",
    ),
    QuizQuestion(
        question_id="q_sys_03",
        topic_id="system_design",
        text="What is the 'Cache Stampede' (or thundering herd) problem, and how is it mitigated?",
        options=[
            "When the cache memory runs out and drops the operating system kernel.",
            "When a popular cached key expires and hundreds of concurrent requests simultaneously hit the database; mitigated via mutex locking or probabilistic early refresh.",
            "When invalid data is written into the cache permanently.",
            "When a client sends malformed JSON payloads to Redis."
        ],
        correct_index=1,
        explanation="When a hot key expires, simultaneous incoming queries overwhelm the database. Using distributed locks or probabilistic early recomputation protects the DB.",
        difficulty="intermediate",
    ),
    QuizQuestion(
        question_id="q_sys_04",
        topic_id="system_design",
        text="In the Circuit Breaker pattern for distributed microservices, what triggers the transition from 'Open' to 'Half-Open'?",
        options=[
            "A sudden spike in inbound request volume.",
            "An expiration timeout period elapses, allowing a trial percentage of requests through to test if the failing downstream service has recovered.",
            "A manual administrator terminal restart command.",
            "When the database runs out of available connection pools."
        ],
        correct_index=1,
        explanation="After entering Open state to fail-fast, a timer lapses transitioning to Half-Open, allowing limited probe requests to verify if the downstream system has healed.",
        difficulty="advanced",
    ),
]

QUESTIONS_BY_TOPIC: Dict[str, List[QuizQuestion]] = {}
for q in QUESTION_BANK:
    QUESTIONS_BY_TOPIC.setdefault(q.topic_id, []).append(q)

QUESTIONS_BY_ID: Dict[str, QuizQuestion] = {q.question_id: q for q in QUESTION_BANK}
