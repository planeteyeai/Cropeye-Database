from typing import Dict, Any
import requests
from datetime import datetime
import ee
import numpy as np
import math
from db import get_connection
import json

# ------------------------------
# Helpers
# ------------------------------

def _is_num(x):
    return isinstance(x, (int, float, np.number))

def _round_safe(x, nd=4):
    if x is None:
        return None
    try:
        xf = float(x)
        if not math.isfinite(xf):
            return None
        return round(xf, nd)
    except Exception:
        return None

def _clean_numbers(obj):
    if isinstance(obj, dict):
        return {k: _clean_numbers(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_numbers(v) for v in obj]
    if _is_num(obj):
        try:
            xf = float(obj)
            return xf if math.isfinite(xf) else None
        except Exception:
            return None
    return obj

def strip_z(coords):
    if not isinstance(coords, list) or not coords:
        return coords

    if isinstance(coords[0], list):
        if isinstance(coords[0][0], list):
            return [strip_z(c) for c in coords]
        else:
            return [[pt[0], pt[1]] for pt in coords if len(pt) >= 2]
    else:
        if len(coords) >= 2:
            return [coords[0], coords[1]]
        return coords

# ------------------------------
# Plot Sync Service
# ------------------------------

class PlotSyncService:

    def __init__(self, django_api_url: str = "https://cropeye-backendd.up.railway.app"):
        self.django_api_url = django_api_url
        self.plots_cache = {}
        self.last_sync = None
        self.cache_duration = 300

    def fetch_plots_from_api(self) -> Dict[str, Any]:
        """Fetch ALL plots with pagination"""

        all_results = []
        url = f"{self.django_api_url}/api/plots/public/"

        try:
            while url:
                response = requests.get(
                    url,
                    headers={'Content-Type': 'application/json'},
                    timeout=180
                )

                if response.status_code != 200:
                    print(f"❌ Django API error: {response.status_code}", flush=True)
                    break

                data = response.json()

                results = data.get("results", [])
                all_results.extend(results)

                # pagination
                url = data.get("next")

            print(f"✅ Total plots fetched: {len(all_results)}", flush=True)

            return self._process_plots_response({"results": all_results})

        except Exception as e:
            print(f"❌ API fetch failed: {e}", flush=True)
            return {}

    def _process_plots_response(self, plots_data: Dict[str, Any]) -> Dict[str, Dict]:
        plot_dict = {}

        for plot in plots_data.get('results', []):

            plot_id = plot.get('id')
            gat_number = plot.get('gat_number', '')
            plot_number = plot.get('plot_number', '')

            address = plot.get('address', {})
            village = address.get('village', '')

            farms = plot.get('farms', [])
            plantation_date = None
            plantation_type = None

            crop_type_name = plot.get('crop_type_name')

            if crop_type_name is None and isinstance(plot.get('crop_type'), dict):
                crop_type_name = plot.get('crop_type', {}).get('name')

            if farms:
                plantation_date = farms[0].get('plantation_date')
                plantation_type = farms[0].get('plantation_type')

                if crop_type_name is None:
                    crop_type_name = farms[0].get('crop_type_name')

                if crop_type_name is None and isinstance(farms[0].get('crop_type'), dict):
                    crop_type_name = farms[0].get('crop_type', {}).get('name')

            plot_name = (
                f"{gat_number}_{plot_number}"
                if gat_number and plot_number
                else gat_number or f"plot_{plot_id}"
            )

            boundary = plot.get('boundary')
            geometry, coords = None, None

            if isinstance(boundary, dict) and boundary.get('coordinates'):
                coords = strip_z(boundary['coordinates'])
                try:
                    geometry = ee.Geometry.Polygon(coords)
                except Exception as e:
                    print(f"❌ Geometry error {plot_id}: {e}", flush=True)
                    continue

            elif plot.get('location') and plot['location'].get('coordinates'):
                location = plot['location']['coordinates']
                lat, lng = location[1], location[0]

                offset = 0.001
                polygon_coords = [[
                    [lng - offset, lat - offset],
                    [lng + offset, lat - offset],
                    [lng + offset, lat + offset],
                    [lng - offset, lat + offset],
                    [lng - offset, lat - offset]
                ]]

                coords = strip_z(polygon_coords)
                geometry = ee.Geometry.Polygon(coords)

            else:
                continue

            plot_dict[plot_name] = {
                "geometry": geometry,
                "geom_type": "Polygon",
                "original_coords": coords,
                "properties": {
                    "django_id": plot_id,
                    "plantation_date": plantation_date,
                    "plantation_type": plantation_type,
                    "crop_type_name": crop_type_name,
                    "village": village,
                    "taluka": address.get('taluka', ''),
                    "district": address.get('district', ''),
                }
            }

        return plot_dict

    def get_plots_dict(self, force_refresh: bool = False) -> Dict[str, Dict]:

        now = datetime.now()

        if (
            not force_refresh and
            self.last_sync and
            (now - self.last_sync).seconds < self.cache_duration and
            self.plots_cache
        ):
            return self.plots_cache

        plots = self.fetch_plots_from_api()

        self.plots_cache = plots
        self.last_sync = now

        return plots

# ------------------------------
# DB SYNC
# ------------------------------

def run_plot_sync():

    print("🔄 Starting internal plot sync...", flush=True)

    plot_service = PlotSyncService()
    plot_dict = plot_service.get_plots_dict(force_refresh=True)

    conn = get_connection()
    cursor = conn.cursor()

    try:
        for name, data in plot_dict.items():

            try:
                geom = data.get("geometry")
                if not geom:
                    continue

                geom_geojson = geom.getInfo()
                area_ha = float(geom.area().divide(10000).getInfo())

                props = data.get("properties", {})

                cursor.execute(
                    """
                    INSERT INTO plots
                    (plot_name, geojson, area_hectares, django_plot_id, plantation_date, crop_type)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (plot_name)
                    DO UPDATE SET
                        geojson = EXCLUDED.geojson,
                        area_hectares = EXCLUDED.area_hectares,
                        django_plot_id = EXCLUDED.django_plot_id,
                        plantation_date = EXCLUDED.plantation_date,
                        crop_type = EXCLUDED.crop_type
                    """,
                    (
                        name,
                        json.dumps(geom_geojson),
                        area_ha,
                        str(props.get("django_id")),
                        props.get("plantation_date"),
                        props.get("crop_type_name"),
                    )
                )

                conn.commit()

            except Exception as e:
                conn.rollback()
                print(f"❌ Error inserting {name}: {e}", flush=True)

    finally:
        cursor.close()
        conn.close()

    print("✅ Internal sync complete", flush=True)

# ------------------------------
# BACKFILL
# ------------------------------

def trigger_new_plot_backfill():

    plot_service = PlotSyncService()
    plots = plot_service.get_plots_dict(force_refresh=True)

    conn = get_connection()
    cursor = conn.cursor()

    try:
        for plot_name, plot_data in plots.items():

            try:
                cursor.execute(
                    "SELECT id, backfill_completed FROM plots WHERE plot_name=%s",
                    (plot_name,)
                )

                row = cursor.fetchone()

                if not row:
                    continue

                plot_id = row[0]
                backfill_done = row[1]

                if backfill_done:
                    continue

                print(f"🆕 Running backfill: {plot_name}", flush=True)

                from Admin import run_monthly_backfill_for_plot
                run_monthly_backfill_for_plot(plot_name, plot_data)

                cursor.execute(
                    "UPDATE plots SET backfill_completed=TRUE WHERE id=%s",
                    (plot_id,)
                )

                conn.commit()

            except Exception as e:
                conn.rollback()
                print(f"🔥 Backfill failed: {plot_name} -> {e}", flush=True)

    finally:
        cursor.close()
        conn.close()
