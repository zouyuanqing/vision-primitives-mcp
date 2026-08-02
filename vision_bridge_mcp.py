"""兼容入口（deprecated）。

项目已更名为 vision-primitives-mcp，主文件为 vision_primitives_mcp.py。
本文件保留用于兼容旧配置路径（vision_bridge_mcp.py）与旧 import，
功能与主文件完全一致。
"""
from vision_primitives_mcp import *  # noqa: F401,F403
from vision_primitives_mcp import main  # noqa: F401

if __name__ == "__main__":
    main()
