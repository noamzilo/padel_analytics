#!/usr/bin/env python
"""
Padel-analytics CLI — functional equivalent of the original Streamlit dashboard.
Reads data.csv, draws trajectories on a 2-D padel court and plots speed-norm
time-series for up to four players.

Run: python padel_cli.py [--html] [--png] [--csv PATH]

Author: OpenAI chat-assistant
"""

import argparse
import pathlib
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, List

# --------------------------------------------------------------------------- #
#                             Court background helper                         #
# --------------------------------------------------------------------------- #
try:
	from visualizations.padel_court import padel_court_2d	# noqa: E402
	USE_BUILTIN_COURT = True
except ModuleNotFoundError:
	USE_BUILTIN_COURT = False


def fallback_court_shape() -> List[go.Scatter]:
	"""
	Returns a list with one Plotly scatter that draws a simple 20 m × 10 m padel
	court rectangle (scale = metres).
	"""
	outline_x = [-5, 5, 5, -5, -5]
	outline_y = [-10, -10, 10, 10, -10]
	return [
		go.Scatter(
			x=outline_x,
			y=outline_y,
			mode="lines",
			line=dict(width=3),
			name="Court outline",
			showlegend=False,
		)
	]


# --------------------------------------------------------------------------- #
#                          Figure-building utilities                          #
# --------------------------------------------------------------------------- #
def make_court_figure(dataframe: pd.DataFrame) -> go.Figure:
	"""
	Returns a Plotly Figure with court background and XY trajectories for each
	player present in the CSV.
	"""
	figure = go.Figure()

	# draw court
	if USE_BUILTIN_COURT:
		for trace in padel_court_2d().data:
			figure.add_trace(trace)
	else:
		for trace in fallback_court_shape():
			figure.add_trace(trace)

	# player trajectories
	player_columns = [
		col.removesuffix("_x") for col in dataframe.columns if col.endswith("_x")
	]
	for player_column_prefix in player_columns:
		x_column = f"{player_column_prefix}_x"
		y_column = f"{player_column_prefix}_y"
		if x_column not in dataframe.columns or y_column not in dataframe.columns:
			continue
		figure.add_trace(
			go.Scatter(
				x=dataframe[x_column],
				y=dataframe[y_column],
				mode="lines+markers",
				name=player_column_prefix.replace("_", " ").title(),
				marker=dict(size=6),
			)
		)

	figure.update_layout(
		title="Player trajectories on padel court",
		xaxis=dict(title="X position [m]"),
		yaxis=dict(
			title="Y position [m]",
			scaleanchor="x",
			scaleratio=1,
		),
		template="plotly_white",
		legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
		width=900,
		height=600,
	)
	return figure


def make_velocity_figure(dataframe: pd.DataFrame) -> go.Figure:
	"""
	Returns a Plotly Figure with speed-norm (player*_Vnorm1) versus time.
	If the *_Vnorm1 columns are missing, the figure is empty.
	"""
	figure = go.Figure()
	time_column = "time" if "time" in dataframe.columns else "frame"

	for player_index in range(1, 5):
		vnorm_column = f"player{player_index}_Vnorm1"
		if vnorm_column not in dataframe.columns:
			continue
		figure.add_trace(
			go.Scatter(
				x=dataframe[time_column],
				y=dataframe[vnorm_column],
				mode="lines",
				name=f"Player {player_index}",
			)
		)

	figure.update_layout(
		title="Speed norm over time (Δt = delta_time1)",
		xaxis=dict(title=time_column),
		yaxis=dict(title="Speed [m/s]"),
		template="plotly_white",
		legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
		width=900,
		height=500,
	)
	return figure


# --------------------------------------------------------------------------- #
#                                Main entry-point                             #
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Render padel analytics figures from data.csv"
	)
	parser.add_argument(
		"--csv",
		type=pathlib.Path,
		default=pathlib.Path(__file__).with_name("data.csv"),
		help="Path to CSV (default: data.csv next to script)",
	)
	parser.add_argument(
		"--html",
		action="store_true",
		help="Write court.html and velocity.html instead of opening figure windows",
	)
	parser.add_argument(
		"--png",
		action="store_true",
		help="Write court.png and velocity.png (requires kaleido)",
	)
	return parser


def main() -> None:
	parser = build_parser()
	args = parser.parse_args()

	if not args.csv.exists():
		print(f"❌  CSV not found: {args.csv}", file=sys.stderr)
		sys.exit(1)

	dataframe = pd.read_csv(args.csv)

	court_figure = make_court_figure(dataframe)
	velocity_figure = make_velocity_figure(dataframe)

	if args.html:
		court_figure.write_html("court.html", include_plotlyjs="cdn")
		velocity_figure.write_html("velocity.html", include_plotlyjs="cdn")
		print("✅  Saved court.html and velocity.html")

	if args.png:
		try:
			court_figure.write_image("court.png", width=1600, height=900, scale=2)
			velocity_figure.write_image("velocity.png", width=1600, height=900, scale=2)
			print("✅  Saved court.png and velocity.png")
		except ValueError:
			print("⚠️  To export PNGs install kaleido:  pip install -U kaleido", file=sys.stderr)

	if not args.html and not args.png:
		# interactive windows
		court_figure.show()
		velocity_figure.show()


if __name__ == "__main__":
	main()
