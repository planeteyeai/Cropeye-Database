from datetime import date, timedelta
from gee_growth import run_growth_analysis_by_plot
from shared_services import PlotSyncService
from db import supabase

print("🚀 DAILY GROWTH WORKER STARTED")

today = date.today().isoformat()
start_date = (date.today() - timedelta(days=30)).isoformat()

plots = PlotSyncService().get_plots_dict(force_refresh=True)

for plot_name, plot_data in plots.items():
    print("Processing", plot_name)
    # same logic you already have
