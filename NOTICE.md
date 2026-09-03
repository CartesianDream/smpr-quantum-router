# Notices and provenance

SMPR is the public name of the Safe Multi-Layout Portfolio Router. This package
contains the executable routing core only. Historical exploration scripts and
their original identifiers are deliberately excluded and are distributed in a
separate reproducibility archive.

The routing policy was developed through evaluation-driven search and manual
verification. The research workflow used SimpleTES as a discovery framework.
The SimpleTES project states that programs discovered by the framework are not
automatically licensed under the framework's AGPL solely because the framework
found them. This package does not include the SimpleTES framework. Cite:

- W. Wang et al., “Evaluation-driven Scaling for Scientific Discovery,”
  arXiv:2604.19341, 2026.
- SimpleTES source: https://github.com/wq-will/SimpleTES

The optional V7 external-validation workflow downloads a pinned subset of
QASMBench. QASMBench is distributed under a BSD license and must be cited:

- A. Li, S. Stein, S. Krishnamoorthy, and J. Ang, “QASMBench: A Low-Level
  Quantum Benchmark Suite for NISQ Evaluation and Simulation,” ACM
  Transactions on Quantum Computing, 2022, DOI: 10.1145/3550488.
- Source: https://github.com/pnnl/QASMBench

Qiskit is used for circuit representation, LightSABRE, OpenQASM/QPY I/O, and
heavy-hex coupling maps. Qiskit and all Python dependencies retain their own
licenses.

