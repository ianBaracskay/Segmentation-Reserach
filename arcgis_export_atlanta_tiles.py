"""
Download Atlanta basemap imagery with an explicit zoom level and verify output.

This script transforms UTM bounds (EPSG:26916) to Web Mercator (EPSG:3857),
counts expected source XYZ tiles at zoom 19, downloads a raster with
contextily.bounds2raster, and reports output resolution/dimensions.
"""

from __future__ import annotations

from pathlib import Path

import contextily as ctx
import mercantile
import rasterio
from pyproj import Transformer


# Your original UTM bounds (EPSG:26916)
LEFT = 738299.3766
BOTTOM = 3735629.2001
RIGHT = 744953.2947
TOP = 3743524.6855

# Explicit zoom to avoid implicit contextily zoom selection.
ZOOM = 19

# Output raster path.
OUTPUT_RASTER = Path(r"C:\Tiles\Atlanta_base_z19.tif")


def main() -> int:
    OUTPUT_RASTER.parent.mkdir(parents=True, exist_ok=True)

    transformer = Transformer.from_crs("EPSG:26916", "EPSG:3857", always_xy=True)
    left_m, bottom_m = transformer.transform(LEFT, BOTTOM)
    right_m, top_m = transformer.transform(RIGHT, TOP)

    # Check what zoom-19 source tile coverage looks like for these bounds.
    tiles = list(mercantile.tiles(left_m, bottom_m, right_m, top_m, zooms=ZOOM))
    print(f"Number of source tiles at zoom {ZOOM}: {len(tiles)}")

    # Download basemap with explicit zoom.
    ctx.bounds2raster(
        left_m,
        bottom_m,
        right_m,
        top_m,
        path=str(OUTPUT_RASTER),
        source=ctx.providers.Esri.WorldImagery,
        zoom=ZOOM,
    )

    # Verify what was written.
    with rasterio.open(OUTPUT_RASTER) as src:
        print(f"Actual resolution: {src.res[0]:.4f}m/px")
        print(f"Expected tiles at 2000px: {src.width // 2000} cols x {src.height // 2000} rows")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
