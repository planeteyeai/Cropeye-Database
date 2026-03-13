from typing import Dict, Any
import requests
from datetime import datetime
import ee
import numpy as np
import math
from db import get_connection
import json
# ------------------------------
# Helpers: safe rounding & JSON sanitization
# ------------------------------

def _is_num(x):
    return isinstance(x, (int, float, np.number))

def _round_safe(x, nd=4):
    """Round numbers safely; pass None through; convert NaN/inf to None."""
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
    """
    Recursively replace NaN/inf with None so JSON is valid.
    Also ensures nested dicts/lists are cleaned.
    """
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
    """Strip Z coordinate from coordinates, handling different coordinate formats"""
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
        else:
            return coords


# ------------------------------
# PlotSyncService for Django API
# ------------------------------

class PlotSyncService:
    """
    Service to fetch plot data from Django /plots/ API
    """
    
    def __init__(self, django_api_url: str = "https://cropeye-backend.up.railway.app/"):
        self.django_api_url = django_api_url
        self.plots_cache = {}
        self.last_sync = None
        self.cache_duration = 300  # 5 minutes cache

    def fetch_plots_from_api(self) -> Dict[str, Any]:
        """Fetch all plots from Django API"""
        try:
            response = requests.get(
                f"{self.django_api_url}/api/plots/public/",
                headers={'Content-Type': 'application/json'},
                timeout=180
            )

            if response.status_code == 200:
                plots_data = response.json()
                return self._process_plots_response(plots_data)
            else:
                print(f"Warning: Django API returned status {response.status_code}. Using empty plot list.")
                return {}

        except requests.exceptions.RequestException as e:
            print(f"Warning: Could not connect to Django API: {str(e)}. Using empty plot list.")
            return {}

    def _process_plots_response(self, plots_data: Dict[str, Any]) -> Dict[str, Dict]:
        """Process the Django API response and convert to plot dictionary format"""
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

            plot_name = f"{gat_number}_{plot_number}" if gat_number and plot_number else gat_number or f"plot_{plot_id}"

            geometry, geom_type, coords = None, None, None
            boundary = plot.get('boundary')

            if isinstance(boundary, dict) and 'coordinates' in boundary and boundary['coordinates']:
                coords = strip_z(boundary['coordinates'])
                geom_type = 'Polygon'
                try:
                    geometry = ee.Geometry.Polygon(coords)
                except Exception as e:
                    print(f"Error creating Polygon geometry for plot {plot_id}: {e}")
                    continue
            elif plot.get('location') and plot['location'].get('coordinates'):
                location = plot['location']['coordinates']
                lat, lng = location[1], location[0]
                offset = 0.001
                polygon_coords = [[
                    [lng - offset, lat - offset], [lng + offset, lat - offset],
                    [lng + offset, lat + offset], [lng - offset, lat + offset],
                    [lng - offset, lat - offset]
                ]]
                coords = strip_z(polygon_coords)
                geom_type = 'Polygon'
                geometry = ee.Geometry.Polygon(coords)
            else:
                print(f"Warning: Plot {plot_id} has no valid geometry, skipping.")
                continue

            plot_dict[plot_name] = {
                "geometry": geometry,
                "geom_type": geom_type,
                "original_coords": coords,
                "properties": {
                    'gat_number': gat_number,
                    'plot_number': plot_number,
                    'village': village,
                    'taluka': address.get('taluka', ''),
                    'district': address.get('district', ''),
                    'state': address.get('state', ''),
                    'country': address.get('country', ''),
                    'pin_code': address.get('pin_code', ''),
                    'django_id': plot_id,
                    'plantation_date': plantation_date,
                    'plantation_type': plantation_type,
                    'crop_type_name': crop_type_name,
                }
            }

        return plot_dict

    def get_plots_dict(self, force_refresh: bool = False) -> Dict[str, Dict]:
        """Get plots dictionary, with caching"""
        current_time = datetime.now()

        if (not force_refresh and self.last_sync and 
            (current_time - self.last_sync).seconds < self.cache_duration and 
            self.plots_cache):
            return self.plots_cache

        plots_data = self.fetch_plots_from_api()
        self.plots_cache = plots_data
        self.last_sync = current_time
        return plots_data


def run_plot_sync():

    print("🔄 Starting internal plot sync...", flush=True)

    inserted = 0
    skipped = 0
    errors = 0
    processed = 0

    plot_service = PlotSyncService()
    plot_dict = plot_service.get_plots_dict(force_refresh=True)

    conn = get_connection()
    cursor = conn.cursor()

    try:

        cursor.execute("SELECT plot_name FROM plots")
        rows = cursor.fetchall()

        existing_names = {r[0] for r in rows}

        for name, data in plot_dict.items():

            processed += 1

            try:

                geom = data.get("geometry")

                if not geom:
                    print(f"⚠ Plot {name} has no valid geometry, skipping.", flush=True)
                    skipped += 1
                    continue

                geom_geojson = geom.getInfo()
                area_ha = float(geom.area().divide(10000).getInfo())

                props = data.get("properties", {})

                django_id = props.get("django_id")
                plantation_date = props.get("plantation_date")
                crop_type = props.get("crop_type_name")

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
                        str(django_id) if django_id else None,
                        plantation_date,
                        crop_type
                    )
                )

                conn.commit()
                inserted += 1

            except Exception as e:

                conn.rollback()
                errors += 1
                print(f"❌ Error inserting {name}: {e}", flush=True)

    finally:

        cursor.close()
        conn.close()

    print("✅ Internal sync complete", flush=True)

    return {
        "inserted": inserted,
        "skipped": skipped,
        "errors": errors,
        "processed": processed
    }


def trigger_new_plot_backfill():
    """
    Detect new plots and run backfill immediately
    """

    plot_service = PlotSyncService()
    plots = plot_service.get_plots_dict(force_refresh=True)

    conn = get_connection()
    cursor = conn.cursor()

    try:

        for plot_name, plot_data in plots.items():

            try:

                cursor.execute(
                    """
                    SELECT id, backfill_completed, last_backfill_hash
                    FROM plots
                    WHERE plot_name = %s
                    """,
                    (plot_name,)
                )

                row = cursor.fetchone()

                if not row:
                    continue

                plot_id = row[0]
                backfill_done = row[1]

                if backfill_done:
                    continue

                props = plot_data.get("properties", {})

                plantation_date = props.get("plantation_date")
                crop_type = props.get("crop_type_name")

                new_hash = f"{plantation_date}_{crop_type}"

                print(f"🆕 New plot detected → running backfill: {plot_name}", flush=True)

                run_monthly_backfill_for_plot(plot_name, plot_data)

                cursor.execute(
                    """
                    UPDATE plots
                    SET backfill_completed = TRUE,
                        last_backfill_hash = %s
                    WHERE id = %s
                    """,
                    (new_hash, plot_id)
                )

                conn.commit()

                print(f"✅ Backfill completed for {plot_name}", flush=True)

            except Exception as e:

                conn.rollback()
                print(f"🔥 Backfill failed for {plot_name}: {e}", flush=True)

    finally:

        cursor.close()
        conn.close()
