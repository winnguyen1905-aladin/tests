#!/usr/bin/env python3
"""
Xóa toàn bộ dữ liệu bảng SAM3 (trees, tree_evidences) và ép cột global_vector -> halfvec(384).

Nếu DB chưa có bảng: tạo schema (create_tables) rồi TRUNCATE (rỗng) và migrate.

Chỉ dùng trên server test.

Chạy từ thư mục gốc repo:
  PYTHONPATH=. python3 scripts/reset_postgres_test_data.py

Dùng cùng biến môi trường / .env như ứng dụng (POSTGRES_*).
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sqlalchemy import inspect, text  # noqa: E402


def main() -> int:
    from src.config.appConfig import get_config, load_from_env
    from src.repository.databaseManager import (
        init_db,
        migrate_vector_dimension,
        shutdown_db,
    )
    # Đăng ký metadata: phải import class, không chỉ Base (tránh create_all rỗng trên một số bản fork).
    from src.repository.entityModels import Base, Tree, TreeEvidence  # noqa: F401

    load_from_env()
    cfg = get_config()
    manager = init_db(app_config=cfg)
    if not manager.is_connected or manager._engine is None:
        print("Không kết nối được PostgreSQL.", file=sys.stderr)
        return 1

    assert manager._engine is not None
    Base.metadata.create_all(manager._engine, checkfirst=True)

    insp = inspect(manager._engine)
    if not insp.has_table("tree_evidences") or not insp.has_table("trees"):
        names = ", ".join(insp.get_table_names()) or "(không có bảng nào)"
        print(
            "Không tạo được bảng trees / tree_evidences. "
            f"Bảng hiện có trong DB: {names}. "
            "Kiểm tra POSTGRES_* và đúng database.",
            file=sys.stderr,
        )
        shutdown_db()
        return 1

    with manager._engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE tree_evidences, trees CASCADE"))

    if not migrate_vector_dimension(384, recreate_hnsw=True):
        print("migrate_vector_dimension(384) thất bại.", file=sys.stderr)
        shutdown_db()
        return 1

    shutdown_db()
    print("OK: schema (nếu thiếu), TRUNCATE trees + tree_evidences, global_vector -> halfvec(384)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
