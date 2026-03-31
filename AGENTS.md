# SAM3

Production-ready tree identification and matching system built around a hierarchical computer-vision pipeline for durian trees.

## Core Stack

- FastAPI API server
- Milvus for global embedding search
- MinIO for descriptor/object storage
- PostgreSQL for metadata and tree records
- Docker Compose for local orchestration

## Matching Pipeline

1. Milvus search with DINO embeddings for coarse candidate retrieval.
2. Angular clustering to keep spatially coherent candidates.
3. K-hop neighbor expansion for neighborhood-aware scoring.
4. SuperPoint plus LightGlue for fine matching.
5. RANSAC for geometric verification.
6. Weighted fusion with DINO as the dominant signal.

## Working Rules

- Preserve the hierarchical pipeline semantics when changing ranking, filtering, or scoring logic.
- Keep DTO changes aligned with FastAPI `response_model` and request validation.
- Follow the existing `dependency-injector` container pattern in `src/repository/containers.py`.
- For storage or retrieval changes, consider Milvus, MinIO, and PostgreSQL impacts together.
- Prefer targeted verification after changes: unit tests first, then integration or service-level checks when needed.

## Key Paths

- `main.py` for the FastAPI entry point
- `src/dto/` for request and response models
- `src/service/` for business logic
- `src/processor/` for CV processing stages
- `src/repository/` for DI and persistence adapters
- `src/config/` for configuration and settings

## Useful Commands

- `docker-compose up -d`
- `python main.py`
- `pytest tests/ -v`
- `make test-unit`
- `make test-integration`
- `make typecheck`

## More Context

See `.claude/CLAUDE.md` for the fuller project notes, deployment details, and CI/CD workflow.

| Action       | Ý nghĩa                                             |
| ------------ | --------------------------------------------------- |
| **feat**     | Thêm tính năng mới                                  |
| **fix**      | Sửa bug                                             |
| **docs**     | Thay đổi tài liệu                                   |
| **style**    | Format code, whitespace, prettier (không đổi logic) |
| **refactor** | Sửa code nhưng không đổi chức năng                  |
| **perf**     | Tối ưu hiệu năng                                    |
| **test**     | Thêm/sửa test                                       |
| **build**    | Thay đổi build system, dependencies                 |
| **ci**       | Thay đổi CI/CD                                      |
| **chore**    | Việc lặt vặt (rename file, update config...)        |
| **revert**   | Revert commit                                       |
| **init**     | Commit đầu tiên                                     |
| **merge**    | Merge branch                                        |
