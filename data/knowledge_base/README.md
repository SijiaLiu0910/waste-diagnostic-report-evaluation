# Regulatory knowledge base

| File | Contents |
|---|---|
| `documents.csv` | Titles, document types, applicable provisions, and official links for 16 regulatory documents |
| `clause_catalog.csv` | Document titles, article or section numbers, clause IDs, and source links for 140 entries |
| `clause_examples.csv` | Three complete clause examples, including clause text and necessary context |

See [data availability](../../DATA_AVAILABILITY.md) for access to the complete clause text table. For formal retrieval, set `CLAUSE_CSV` to the complete table, using the same columns as `clause_examples.csv` and containing 140 records.

The example table illustrates the data structure. To run a demonstration with this table, also set `ALLOW_SAMPLE_CORPUS=1`. The example corpus is not used for formal retrieval evaluation.
