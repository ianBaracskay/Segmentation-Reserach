"""
Export Atlanta imagery into uniform high-resolution GeoTIFF tiles using ArcPy.

Run from ArcGIS Pro's Python environment (Python Command Prompt) so arcpy is available.

What this script does:
1) Validates source raster/layer and city boundary.
2) Builds a fishnet grid over Atlanta.
3) Keeps only tiles that intersect the boundary.
4) Clips source imagery into one GeoTIFF per tile.
5) Writes tile metadata CSV with extents.

Notes:
- For best results, use a projected CRS with meter units.
- If your source units are feet, adjust TILE_SIZE_UNITS and OVERLAP_UNITS accordingly.
"""

from __future__ import annotations

import csv
import os
import sys
import traceback
from dataclasses import dataclass

import arcpy


# ----------------------------
# USER CONFIG
# ----------------------------

# Source imagery (raster dataset path OR raster layer name in current Pro project)
# Examples:
#   r"C:/data/atlanta_imagery.tif"
#   "Atlanta_Imagery_Layer"
SOURCE_RASTER = r"C:/data/atlanta_imagery.tif"

# City boundary polygon feature class/shapefile for Atlanta.
# Example:
#   r"C:/data/atlanta_boundary.gdb/atlanta_city_boundary"
CITY_BOUNDARY = r"C:/data/atlanta_boundary.shp"

# Output folder for tile GeoTIFFs + metadata CSV.
OUTPUT_FOLDER = r"C:/data/atlanta_tiles"

# Tile size and overlap in source map units.
# If your raster CRS is meters, these are meters.
# 512 with 64 overlap is a good default for urban modeling.
TILE_SIZE_UNITS = 512.0
OVERLAP_UNITS = 64.0

# Keep only tiles that intersect city boundary.
KEEP_ONLY_INTERSECTING = True

# Tile naming prefix.
TILE_PREFIX = "atlanta"

# Compression for output GeoTIFFs.
# Valid values vary by raster type; LZW is a safe default.
TIFF_COMPRESSION = "LZW"

# Overwrite existing outputs.
OVERWRITE_OUTPUT = False


# ----------------------------
# INTERNAL HELPERS
# ----------------------------


@dataclass
class TileRecord:
    tile_id: str
    row: int
    col: int
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    path: str


def _log(msg: str) -> None:
    print(f"[INFO] {msg}")


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def _err(msg: str) -> None:
    print(f"[ERROR] {msg}")


def _assert_exists(path_or_layer: str, label: str) -> None:
    if not arcpy.Exists(path_or_layer):
        raise RuntimeError(f"{label} does not exist or is not accessible: {path_or_layer}")


def _get_spatial_ref(path_or_layer: str) -> arcpy.SpatialReference:
    desc = arcpy.Describe(path_or_layer)
    sr = desc.spatialReference
    if sr is None or sr.name in ("Unknown", "Unknown Coordinate System"):
        raise RuntimeError(f"Spatial reference is unknown for: {path_or_layer}")
    return sr


def _safe_make_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _make_fishnet(
    boundary_fc: str,
    out_fishnet: str,
    tile_size: float,
) -> None:
    desc = arcpy.Describe(boundary_fc)
    ext = desc.extent

    origin_coord = f"{ext.XMin} {ext.YMin}"
    y_axis_coord = f"{ext.XMin} {ext.YMin + 1.0}"
    opposite_corner = f"{ext.XMax} {ext.YMax}"

    arcpy.management.CreateFishnet(
        out_feature_class=out_fishnet,
        origin_coord=origin_coord,
        y_axis_coord=y_axis_coord,
        cell_width=tile_size,
        cell_height=tile_size,
        number_rows="0",
        number_columns="0",
        corner_coord=opposite_corner,
        labels="NO_LABELS",
        template=boundary_fc,
        geometry_type="POLYGON",
    )


def _iter_tiles(fishnet_fc: str):
    fields = ["OID@", "SHAPE@"]
    with arcpy.da.SearchCursor(fishnet_fc, fields) as cursor:
        for oid, shape in cursor:
            yield oid, shape


def _clip_tile(
    source_raster: str,
    tile_geom,
    out_tif: str,
) -> None:
    # Clip by polygon geometry; maintain extent to fixed tile footprint.
    arcpy.management.Clip(
        in_raster=source_raster,
        rectangle="#",
        out_raster=out_tif,
        in_template_dataset=tile_geom,
        nodata_value="#",
        clipping_geometry="ClippingGeometry",
        maintain_clipping_extent="MAINTAIN_EXTENT",
    )


def _set_tiff_compression(compression: str) -> None:
    # Applies to geoprocessing outputs in this session.
    env_value = compression.strip().upper()
    arcpy.env.compression = env_value


def _write_metadata_csv(path: str, rows: list[TileRecord]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["tile_id", "row", "col", "xmin", "ymin", "xmax", "ymax", "file_path"])
        for r in rows:
            writer.writerow([r.tile_id, r.row, r.col, r.xmin, r.ymin, r.xmax, r.ymax, r.path])


def _row_col_from_extent(
    x_min: float,
    y_min: float,
    full_x_min: float,
    full_y_min: float,
    step: float,
) -> tuple[int, int]:
    col = int(round((x_min - full_x_min) / step))
    row = int(round((y_min - full_y_min) / step))
    return row, col


def main() -> int:
    arcpy.env.overwriteOutput = bool(OVERWRITE_OUTPUT)

    _log("Validating inputs")
    _assert_exists(SOURCE_RASTER, "SOURCE_RASTER")
    _assert_exists(CITY_BOUNDARY, "CITY_BOUNDARY")

    if OVERLAP_UNITS < 0:
        raise RuntimeError("OVERLAP_UNITS must be >= 0")
    if TILE_SIZE_UNITS <= 0:
        raise RuntimeError("TILE_SIZE_UNITS must be > 0")
    if OVERLAP_UNITS >= TILE_SIZE_UNITS:
        raise RuntimeError("OVERLAP_UNITS must be smaller than TILE_SIZE_UNITS")

    _safe_make_dir(OUTPUT_FOLDER)

    _log("Inspecting source spatial reference")
    raster_sr = _get_spatial_ref(SOURCE_RASTER)
    _log(f"Source CRS: {raster_sr.name}")
    unit_name = getattr(raster_sr, "linearUnitName", "") or "unknown"
    _log(f"Source linear unit: {unit_name}")

    # Reproject boundary into raster CRS so fishnet aligns exactly with source imagery.
    temp_workspace = "in_memory"
    boundary_projected = os.path.join(temp_workspace, "atl_boundary_proj")
    fishnet_fc = os.path.join(temp_workspace, "atl_fishnet")
    fishnet_filtered = os.path.join(temp_workspace, "atl_fishnet_filtered")

    _log("Projecting city boundary to source CRS")
    arcpy.management.Project(CITY_BOUNDARY, boundary_projected, raster_sr)

    _log("Creating fishnet grid")
    _make_fishnet(boundary_projected, fishnet_fc, TILE_SIZE_UNITS)

    work_fishnet = fishnet_fc
    if KEEP_ONLY_INTERSECTING:
        _log("Filtering fishnet to tiles intersecting city boundary")
        arcpy.analysis.PairwiseIntersect([fishnet_fc, boundary_projected], fishnet_filtered)
        # PairwiseIntersect creates cut geometries; dissolve by original fishnet FID to get one row per tile.
        # For in_memory data, fishnet OID is often represented as FID_atl_fishnet.
        fields = [f.name for f in arcpy.ListFields(fishnet_filtered)]
        fid_field = None
        for cand in fields:
            if cand.lower().startswith("fid_") and "fishnet" in cand.lower():
                fid_field = cand
                break

        if fid_field is None:
            _warn("Could not find fishnet FID field after intersect; using unfiltered fishnet")
        else:
            fishnet_filtered_diss = os.path.join(temp_workspace, "atl_fishnet_filtered_diss")
            arcpy.management.Dissolve(fishnet_filtered, fishnet_filtered_diss, [fid_field])
            # Join back to original fishnet geometry by OID/FID mapping.
            fishnet_selected = os.path.join(temp_workspace, "atl_fishnet_selected")
            arcpy.management.MakeFeatureLayer(fishnet_fc, "fishnet_lyr")
            vals = [r[0] for r in arcpy.da.SearchCursor(fishnet_filtered_diss, [fid_field])]
            if vals:
                oid_field = arcpy.Describe(fishnet_fc).OIDFieldName
                chunk = ",".join(str(int(v)) for v in vals)
                sql = f"{arcpy.AddFieldDelimiters(fishnet_fc, oid_field)} IN ({chunk})"
                arcpy.management.SelectLayerByAttribute("fishnet_lyr", "NEW_SELECTION", sql)
                arcpy.management.CopyFeatures("fishnet_lyr", fishnet_selected)
                work_fishnet = fishnet_selected
            else:
                _warn("No intersecting tiles found. Check city boundary and source CRS.")

    # Metadata prep
    desc_boundary = arcpy.Describe(boundary_projected)
    b_ext = desc_boundary.extent
    step = TILE_SIZE_UNITS - OVERLAP_UNITS

    _set_tiff_compression(TIFF_COMPRESSION)

    _log("Exporting tiles")
    exported: list[TileRecord] = []
    count_total = int(arcpy.management.GetCount(work_fishnet)[0])
    count_done = 0

    for oid, shape in _iter_tiles(work_fishnet):
        ext = shape.extent
        row, col = _row_col_from_extent(ext.XMin, ext.YMin, b_ext.XMin, b_ext.YMin, step)
        tile_id = f"{TILE_PREFIX}_r{row:04d}_c{col:04d}"
        out_tif = os.path.join(OUTPUT_FOLDER, f"{tile_id}.tif")

        if os.path.exists(out_tif) and not OVERWRITE_OUTPUT:
            count_done += 1
            continue

        try:
            _clip_tile(SOURCE_RASTER, shape, out_tif)
            exported.append(
                TileRecord(
                    tile_id=tile_id,
                    row=row,
                    col=col,
                    xmin=ext.XMin,
                    ymin=ext.YMin,
                    xmax=ext.XMax,
                    ymax=ext.YMax,
                    path=out_tif,
                )
            )
        except Exception as exc:
            _warn(f"Tile failed ({tile_id}): {exc}")

        count_done += 1
        if count_done % 25 == 0 or count_done == count_total:
            _log(f"Progress: {count_done}/{count_total}")

    # Write metadata
    metadata_csv = os.path.join(OUTPUT_FOLDER, f"{TILE_PREFIX}_tile_index.csv")
    # Include existing tiles in metadata too
    for file_name in os.listdir(OUTPUT_FOLDER):
        if not file_name.lower().endswith(".tif"):
            continue
        # Only add those not already in exported list
        full_path = os.path.join(OUTPUT_FOLDER, file_name)
        tile_id = os.path.splitext(file_name)[0]
        if any(r.tile_id == tile_id for r in exported):
            continue
        # Extents for pre-existing tiles are not trivial to re-read cheaply here.
        # Keep path row for visibility even if extents are empty.
        exported.append(
            TileRecord(
                tile_id=tile_id,
                row=-1,
                col=-1,
                xmin=float("nan"),
                ymin=float("nan"),
                xmax=float("nan"),
                ymax=float("nan"),
                path=full_path,
            )
        )

    _write_metadata_csv(metadata_csv, exported)

    _log(f"Done. Exported/Indexed tiles: {len(exported)}")
    _log(f"Tiles folder: {OUTPUT_FOLDER}")
    _log(f"Tile index CSV: {metadata_csv}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        _err(str(exc))
        traceback.print_exc()
        raise SystemExit(1)
