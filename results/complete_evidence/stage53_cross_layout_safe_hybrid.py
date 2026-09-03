from __future__ import annotations

from stage52_live_safe_dual import main


if __name__ == "__main__":
    # 0 表示使用全部“SWAP+深度并列最优”的 LightSABRE 唯一布局。
    # 每个布局再用代理筛选 4 个一换位邻居做真实 OAABR 路由。
    main(
        stage=53,
        default_hybrid_layouts=0,
        default_hybrid_neighbors=4,
    )
