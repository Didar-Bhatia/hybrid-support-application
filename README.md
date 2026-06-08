# hybrid-support-application
To improve our support team's efficiency, I developed a hybrid search pipeline for our document base. The workflow is designed to first find answers by tracing the exact relationships between entities in the data, and if it misses, it seamlessly falls back to a semantic text search. 

The project is a hybrid retrieval application designed to handle both FAQ and complex comparison functionalities. The workflow encompasses:

Document Processing: A pipeline that handles multi-format document loading followed by semantic chunking to preserve context boundaries before passing text to the embedding model.

Knowledge Representation: Entity and relationship extraction to populate a Neo4j graph database.

Retrieval Mechanism: Specialized functions, specifically graph rag query and graph retrieval, orchestrate the data fetching. The system attempts a graph-based retrieval first to capture structural and relational context.

Vector Fallback: If the graph retrieval yields insufficient results, a fallback vector similarity search is triggered to ensure an answer is still provided.

NL2SQL Integration: LlamaIndex is utilized to translate natural language into SQL for structured data querying.

Evaluation & UI: The RAGAS framework is integrated for quantitative performance evaluation, and the entire system is surfaced through a Streamlit-based user interface.
