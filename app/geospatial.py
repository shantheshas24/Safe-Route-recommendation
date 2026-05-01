import os
import osmnx as ox
import networkx as nx
import numpy as np
from typing import Dict, Any

from app.risk_analysis import get_risk_penalty, get_risk_penalties_batch, RISK_MULTIPLIER

# --- CONFIGURATION ---
CITY_NAME = "Bangalore, Karnataka, India"
GRAPHML_PATH = "data/bangalore_graph.graphml"
PROJ_GRAPHML_PATH = "data/bangalore_graph_proj.graphml"

LOCATIONS = {
    "Koramangala": (12.9352, 77.6245),
    "Indiranagar": (12.9719, 77.6412),
    "MG Road": (12.9733, 77.6117),
    "Rajarajeshwari Nagar": (12.9230, 77.5185)
}

# Cache objects — loaded once on first request, reused forever
G_PROJ = None
G_BASE = None


def load_graph():
    global G_PROJ, G_BASE
    if G_PROJ is not None:
        return G_PROJ

    os.makedirs("data", exist_ok=True)

    if os.path.exists(PROJ_GRAPHML_PATH):
        G_PROJ = ox.load_graphml(PROJ_GRAPHML_PATH)
        if os.path.exists(GRAPHML_PATH):
            G_BASE = ox.load_graphml(GRAPHML_PATH)
        else:
            G_BASE = ox.project_graph(G_PROJ, to_crs="epsg:4326")
        return G_PROJ

    if os.path.exists(GRAPHML_PATH):
        G_BASE = ox.load_graphml(GRAPHML_PATH)
    else:
        G_BASE = ox.graph_from_place(CITY_NAME, network_type="drive")
        ox.save_graphml(G_BASE, filepath=GRAPHML_PATH)

    if G_BASE.graph.get("crs") != "epsg:4326":
        G_BASE = ox.project_graph(G_BASE, to_crs="epsg:4326")

    G_PROJ = ox.project_graph(G_BASE)
    ox.save_graphml(G_PROJ, filepath=PROJ_GRAPHML_PATH)
    return G_PROJ


def get_point_risk(lat: float, lon: float, hour: int, day: str) -> Dict[str, Any]:
    try:
        raw_score = get_risk_penalty(lat, lon, hour, day)
        display_score = max(0.0, float(raw_score) - 0.5)

        if display_score > 0.8:
            level = "Very High"
        elif display_score > 0.6:
            level = "High"
        elif display_score > 0.4:
            level = "Medium"
        elif display_score > 0.2:
            level = "Low"
        else:
            level = "Very Low"

        return {
            "status": "success",
            "risk_score": round(display_score, 4),
            "level": level,
            "raw_model_score": round(float(raw_score), 4)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def find_safe_route(orig_lat: float, orig_lon: float, dest_lat: float, dest_lon: float,
                    hour: int, day: str):
    load_graph()  # ensures G_PROJ and G_BASE are loaded

    if G_BASE is None or G_PROJ is None:
        return {"status": "error", "message": "Graph not loaded. Restart service."}

    # --- Step 1: Find nearest nodes (fast, uses spatial index) ---
    orig_node = ox.nearest_nodes(G_BASE, orig_lon, orig_lat)
    dest_node = ox.nearest_nodes(G_BASE, dest_lon, dest_lat)

    if orig_node == dest_node:
        return {"status": "error", "message": "Origin and destination are too close or identical."}

    # --- Step 2: Build a small bbox subgraph around the route corridor ---
    # Buffer scales with trip distance so long routes still work correctly.
    lat_span = abs(dest_lat - orig_lat)
    lon_span = abs(dest_lon - orig_lon)
    pad = max(0.03, max(lat_span, lon_span) * 0.5)   # at least ~3 km, up to 50% of span

    north = max(orig_lat, dest_lat) + pad
    south = min(orig_lat, dest_lat) - pad
    east  = max(orig_lon, dest_lon) + pad
    west  = min(orig_lon, dest_lon) - pad

    # Filter using G_BASE which stores coords as lat/lon (y/x)
    bbox_node_set = {
        n for n, d in G_BASE.nodes(data=True)
        if south <= d['y'] <= north and west <= d['x'] <= east
    }

    # If either endpoint fell outside the box (shouldn't happen), fall back to full graph
    if orig_node not in bbox_node_set or dest_node not in bbox_node_set:
        bbox_node_set = set(G_BASE.nodes())

    nodes_in_bbox = list(bbox_node_set)

    # subgraph() returns a VIEW — zero copy cost
    sub_base = G_BASE.subgraph(nodes_in_bbox)
    sub_proj = G_PROJ.subgraph(nodes_in_bbox)

    # --- Step 3: Compute risk scores ONLY for nodes in the subgraph ---
    nodes       = list(sub_base.nodes(data=True))
    lats        = [d['y'] for _, d in nodes]
    lons        = [d['x'] for _, d in nodes]
    node_ids    = [n for n, _ in nodes]
    risk_scores = get_risk_penalties_batch(lats, lons, hour, day)
    node_risks  = dict(zip(node_ids, risk_scores))

    # --- Step 4: Decide time-of-day multiplier ---
    if 20 <= hour <= 23:
        current_multiplier = RISK_MULTIPLIER * 5
    elif 6 <= hour <= 19:
        current_multiplier = 0.1
    else:
        current_multiplier = RISK_MULTIPLIER

    # --- Step 5: Weight function — no graph mutation, no copy ---
    def edge_weight(u, v, d):
        length_m = d.get('length', 0)
        risk     = node_risks.get(u, 0.1)
        return length_m * (1 + risk * (current_multiplier / 100))

    # --- Step 6: Dijkstra on the small subgraph ---
    try:
        route_nodes = nx.shortest_path(sub_proj, orig_node, dest_node, weight=edge_weight)
    except nx.NetworkXNoPath:
        return {"status": "error", "message": "No path could be found between the selected points."}

    # --- Step 7: Build response ---
    route_coords = []
    total_dist   = 0.0
    total_risk   = 0.0

    for i, node_id in enumerate(route_nodes):
        base_node = G_BASE.nodes[node_id]
        route_coords.append((float(base_node['y']), float(base_node['x'])))

        if i < len(route_nodes) - 1:
            u, v      = route_nodes[i], route_nodes[i + 1]
            edge_data = G_BASE.get_edge_data(u, v)
            if isinstance(edge_data, dict):
                data = list(edge_data.values())[0]
            else:
                data = edge_data
            total_dist += data.get('length', 0)
            total_risk += node_risks.get(u, 0.1)

    avg_risk         = total_risk / len(route_nodes) if route_nodes else 0
    adjusted_penalty = max(0.0, avg_risk - 0.5)

    return {
        "status":            "success",
        "route":             route_coords,
        "total_distance_km": round(total_dist / 1000.0, 2),
        "total_penalty":     round(adjusted_penalty, 4)
    }