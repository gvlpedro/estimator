Model: text-embedding-3-small (dim=1536)

BUD-2024-014::AUTH-001  vs  BUD-2023-004::AUTH-001
  expectation: same concept, same sector and stack (OAuth 2.0 backends in fintech, Rails)
  cosine similarity: 0.8795

BUD-2022-003::PAY-001  vs  BUD-2024-009::CHK-001
  expectation: related concept, different vertical and stack (payments: grocery/Node vs fashion/Laravel)
  cosine similarity: 0.7019

BUD-2024-014::AUTH-001  vs  BUD-2025-013::PDM-001
  expectation: unrelated (fintech OAuth backend vs wind-farm predictive maintenance ML)
  cosine similarity: 0.4454

Sanity check (pair 1 > pair 2 > pair 3): PASS