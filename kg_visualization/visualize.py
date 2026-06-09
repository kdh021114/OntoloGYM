from __future__ import annotations

import html
import json
import pickle
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from common.project_config import load_project_config
from common.run_context import record_pipeline_run


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ONTOGEN_ROOT = PROJECT_ROOT / "ontogen"


@dataclass
class LayoutNode:
    id: str
    label: str
    level: int
    node_type: str
    children: list["LayoutNode"] = field(default_factory=list)
    x: float = 0
    y: float = 0


def _load_taxonomy_tree(path: Path):
    if str(ONTOGEN_ROOT) not in sys.path:
        sys.path.insert(0, str(ONTOGEN_ROOT))
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def _node_label(tree_node: Any) -> str:
    synonyms = getattr(tree_node, "synonyms", None)
    if synonyms:
        return str(synonyms[0])
    value = getattr(tree_node, "value", "")
    if isinstance(value, tuple):
        return " ".join(str(part) for part in value)
    return str(value)


def _tree_to_layout_node(tree_node: Any, level: int = 0, parent_is_root: bool = False) -> LayoutNode:
    label = _node_label(tree_node)
    if level == 0:
        node_type = "root"
    elif parent_is_root:
        node_type = "category"
    else:
        node_type = "term"

    layout_node = LayoutNode(
        id=label,
        label=label,
        level=level,
        node_type=node_type,
    )
    children = getattr(tree_node, "children", []) or []
    layout_node.children = [
        _tree_to_layout_node(child, level=level + 1, parent_is_root=(level == 0))
        for child in children
    ]
    return layout_node


def _collect_graph(node: LayoutNode) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nodes = [
        {
            "id": node.id,
            "label": node.label,
            "type": node.node_type,
            "level": node.level,
        }
    ]
    edges = []
    for child in node.children:
        edges.append(
            {
                "source": child.id,
                "target": node.id,
                "relation": "isA",
                "provenance": "ontogen_taxonomy",
            }
        )
        child_nodes, child_edges = _collect_graph(child)
        nodes.extend(child_nodes)
        edges.extend(child_edges)
    return nodes, edges


def _merge_relation_graph(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    relation_graph_path: Path,
    provenance: str,
) -> None:
    if not relation_graph_path.exists():
        return

    relation_graph = json.loads(relation_graph_path.read_text(encoding="utf-8"))
    existing_node_ids = {node["id"] for node in nodes}
    for node in relation_graph.get("nodes", []):
        node_id = str(node.get("id", "")).strip()
        if not node_id or node_id in existing_node_ids:
            continue
        nodes.append(
            {
                "id": node_id,
                "label": node_id,
                "type": node.get("type", "relation_entity"),
                "level": None,
            }
        )
        existing_node_ids.add(node_id)

    for edge in relation_graph.get("edges", []):
        source = str(edge.get("source", "")).strip()
        target = str(edge.get("target", "")).strip()
        relation = str(edge.get("relation", "")).strip()
        if source and target and relation:
            edges.append(
                {
                    "source": source,
                    "target": target,
                    "relation": relation,
                    "provenance": provenance,
                    "confidence": edge.get("confidence"),
                    "evidence_quote": edge.get("evidence_quote", ""),
                }
            )


def _assign_layout(root: LayoutNode) -> tuple[int, int]:
    leaf_index = 0
    max_depth = 0

    def assign(node: LayoutNode) -> None:
        nonlocal leaf_index, max_depth
        max_depth = max(max_depth, node.level)
        if not node.children:
            node.y = leaf_index * 44 + 40
            leaf_index += 1
        else:
            for child in node.children:
                assign(child)
            node.y = sum(child.y for child in node.children) / len(node.children)
        node.x = node.level * 300 + 30

    assign(root)
    width = max(900, (max_depth + 1) * 320)
    height = max(420, leaf_index * 44 + 80)
    return width, height


def _short_label(label: str, max_chars: int) -> str:
    if len(label) <= max_chars:
        return label
    return label[: max_chars - 3].rstrip() + "..."


def _json_for_script(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False).replace("</", "<\\/")


def _counter_summary(items: Counter[str | None], total: int) -> str:
    rows = []
    for key, value in items.most_common():
        label = html.escape(str(key or "unknown"))
        rows.append(
            f'<span class="pill"><span>{label}</span><strong>{value}</strong></span>'
        )
    return "\n".join(rows) if rows else f'<span class="pill"><span>Total</span><strong>{total}</strong></span>'


def _render_svg(root: LayoutNode, max_label_chars: int) -> tuple[str, int, int]:
    width, height = _assign_layout(root)
    lines = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="KG taxonomy graph">',
        '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L0,6 L7,3 z" fill="#7c8794"/></marker></defs>',
    ]

    def draw_edges(node: LayoutNode) -> None:
        for child in node.children:
            lines.append(
                f'<path class="edge" d="M {child.x + 210:.1f} {child.y:.1f} C {child.x + 255:.1f} {child.y:.1f}, {node.x - 45:.1f} {node.y:.1f}, {node.x:.1f} {node.y:.1f}" marker-end="url(#arrow)" />'
            )
            draw_edges(child)

    def draw_nodes(node: LayoutNode) -> None:
        fill = {
            "root": "#111827",
            "category": "#2563eb",
            "term": "#f8fafc",
        }.get(node.node_type, "#f8fafc")
        stroke = "#1d4ed8" if node.node_type == "category" else "#cbd5e1"
        text_fill = "#ffffff" if node.node_type in {"root", "category"} else "#0f172a"
        label = html.escape(_short_label(node.label, max_label_chars))
        title = html.escape(node.label)
        lines.append(f'<g class="node {node.node_type}" transform="translate({node.x:.1f},{node.y - 17:.1f})">')
        lines.append(f"<title>{title}</title>")
        lines.append(f'<rect width="220" height="34" rx="7" fill="{fill}" stroke="{stroke}" />')
        lines.append(f'<text x="12" y="22" fill="{text_fill}">{label}</text>')
        lines.append("</g>")
        for child in node.children:
            draw_nodes(child)

    draw_edges(root)
    draw_nodes(root)
    lines.append("</svg>")
    return "\n".join(lines), width, height


def _render_canvas_html_legacy(
    graph: dict[str, Any],
    taxonomy_root: LayoutNode,
    output_json: Path,
    max_label_chars: int,
) -> str:
    svg, width, height = _render_svg(taxonomy_root, max_label_chars)
    relation_edges = [
        edge for edge in graph["edges"] if str(edge.get("provenance", "")).startswith("relation")
    ]
    relation_counts = Counter(edge.get("relation") for edge in graph["edges"])
    provenance_counts = Counter(edge.get("provenance") for edge in graph["edges"])
    node_type_counts = Counter(node.get("type") for node in graph["nodes"])
    graph_json = _json_for_script(graph)
    relation_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(edge.get('source', ''))}</td>"
        f"<td>{html.escape(edge.get('relation', ''))}</td>"
        f"<td>{html.escape(edge.get('target', ''))}</td>"
        f"<td>{html.escape(str(edge.get('confidence', '') or ''))}</td>"
        f"<td>{html.escape(edge.get('provenance', ''))}</td>"
        "</tr>"
        for edge in relation_edges
    )
    if not relation_rows:
        relation_rows = '<tr><td colspan="5">No relation graph was found.</td></tr>'

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OntoloGYM KG Visualization</title>
  <style>
    :root {{
      --bg: #f6f7fb;
      --panel: #ffffff;
      --ink: #111827;
      --muted: #64748b;
      --line: #d8dee9;
      --accent: #2563eb;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--ink); background: var(--bg); }}
    header {{ padding: 22px 28px 14px; background: var(--panel); border-bottom: 1px solid var(--line); position: sticky; top: 0; z-index: 10; }}
    h1 {{ margin: 0 0 8px; font-size: 24px; letter-spacing: 0; }}
    h2 {{ margin: 0 0 12px; font-size: 17px; }}
    main {{ padding: 20px 28px 36px; display: grid; gap: 18px; }}
    .meta {{ color: var(--muted); font-size: 14px; line-height: 1.45; }}
    .panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 16px; overflow: hidden; }}
    .stats {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }}
    .pill {{ display: inline-flex; align-items: center; gap: 8px; padding: 6px 9px; border: 1px solid #dbe3ef; border-radius: 999px; background: #f8fafc; color: #334155; font-size: 12px; }}
    .pill strong {{ color: #0f172a; }}
    .graph-shell {{ display: grid; grid-template-columns: minmax(0, 1fr) 340px; gap: 14px; min-height: 720px; }}
    .canvas-wrap {{ position: relative; min-height: 720px; border: 1px solid #dbe3ef; border-radius: 8px; overflow: hidden; background: radial-gradient(circle at 20% 0%, #eef6ff 0, #ffffff 34%, #f8fafc 100%); }}
    canvas {{ display: block; width: 100%; height: 720px; }}
    .toolbar {{ position: absolute; left: 12px; top: 12px; right: 12px; display: flex; flex-wrap: wrap; gap: 8px; align-items: center; pointer-events: none; }}
    .toolbar > * {{ pointer-events: auto; }}
    input, select, button {{ height: 34px; border: 1px solid #cbd5e1; border-radius: 6px; background: #ffffff; color: #0f172a; font-size: 13px; padding: 0 10px; }}
    input[type="range"] {{ padding: 0; width: 120px; }}
    button {{ cursor: pointer; font-weight: 600; }}
    button.active {{ background: var(--accent); border-color: var(--accent); color: #ffffff; }}
    .side {{ border: 1px solid #dbe3ef; border-radius: 8px; overflow: hidden; min-height: 720px; display: grid; grid-template-rows: auto 1fr; background: #ffffff; }}
    .side-head {{ padding: 14px; border-bottom: 1px solid #e2e8f0; }}
    .side-body {{ padding: 14px; overflow: auto; }}
    .legend {{ display: grid; gap: 7px; font-size: 13px; }}
    .legend-item {{ display: flex; align-items: center; gap: 8px; }}
    .swatch {{ width: 14px; height: 14px; border-radius: 4px; border: 1px solid rgba(15, 23, 42, .15); flex: none; }}
    .details {{ margin-top: 16px; padding-top: 14px; border-top: 1px solid #e2e8f0; font-size: 13px; line-height: 1.5; }}
    .details h3 {{ margin: 0 0 8px; font-size: 14px; }}
    .kv {{ margin: 7px 0; }}
    .kv strong {{ display: block; color: #475569; font-size: 11px; text-transform: uppercase; letter-spacing: .04em; }}
    .relation-filter {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }}
    .relation-filter label {{ display: inline-flex; gap: 5px; align-items: center; font-size: 12px; border: 1px solid #dbe3ef; border-radius: 999px; padding: 5px 8px; background: #f8fafc; }}
    .relation-filter input {{ height: auto; }}
    .taxonomy-panel {{ max-height: 520px; overflow: auto; }}
    svg {{ min-width: {width}px; min-height: {height}px; }}
    .edge {{ fill: none; stroke: #94a3b8; stroke-width: 1.4; }}
    text {{ font-size: 12px; dominant-baseline: middle; }}
    .table-panel {{ max-height: 520px; overflow: auto; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
    th, td {{ border-bottom: 1px solid #e2e8f0; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f1f5f9; position: sticky; top: 0; z-index: 1; }}
    td:nth-child(1), td:nth-child(3) {{ max-width: 360px; }}
    @media (max-width: 1080px) {{
      .graph-shell {{ grid-template-columns: 1fr; }}
      .side {{ min-height: 420px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>OntoloGYM KG Visualization</h1>
    <div class="meta">Nodes: {len(graph["nodes"])} · Edges: {len(graph["edges"])} · JSON: {html.escape(str(output_json))}</div>
    <div class="stats">
      {_counter_summary(relation_counts, len(graph["edges"]))}
    </div>
  </header>
  <main>
    <section class="panel">
      <h2>Interactive Relation Graph</h2>
      <div class="meta">Edge labels are drawn on the graph. Search a node, filter relation types, click nodes/edges for evidence and provenance.</div>
      <div class="graph-shell">
        <div class="canvas-wrap">
          <canvas id="kgCanvas"></canvas>
          <div class="toolbar">
            <input id="searchBox" placeholder="Search node or relation">
            <select id="graphMode" title="Graph mode">
              <option value="relation">relation edges</option>
              <option value="taxonomy">isA taxonomy</option>
              <option value="all">all edges</option>
            </select>
            <button id="toggleLabels" class="active" type="button">edge labels</button>
            <button id="fitGraph" type="button">fit</button>
            <label class="pill">max nodes <input id="maxNodes" type="range" min="80" max="700" value="260" step="20"><strong id="maxNodesValue">260</strong></label>
          </div>
        </div>
        <aside class="side">
          <div class="side-head">
            <h2>Legend & Details</h2>
            <div class="meta">Relation counts and selected item metadata.</div>
          </div>
          <div class="side-body">
            <div id="legend" class="legend"></div>
            <div id="relationFilters" class="relation-filter"></div>
            <div id="details" class="details">Click a node or edge to inspect it.</div>
            <div class="details">
              <h3>Node Types</h3>
              <div class="stats">{_counter_summary(node_type_counts, len(graph["nodes"]))}</div>
            </div>
            <div class="details">
              <h3>Provenance</h3>
              <div class="stats">{_counter_summary(provenance_counts, len(graph["edges"]))}</div>
            </div>
          </div>
        </aside>
      </div>
    </section>
    <section class="panel">
      <h2>Ontology Taxonomy Tree</h2>
      <div class="meta">Static isA hierarchy from OntoGen. Use the interactive graph above for relation labels and evidence.</div>
      <div class="taxonomy-panel">{svg}</div>
    </section>
    <section class="panel table-panel">
      <h2>Relation Edge Table</h2>
      <table>
        <thead><tr><th>Source</th><th>Relation</th><th>Target</th><th>Confidence</th><th>Provenance</th></tr></thead>
        <tbody>{relation_rows}</tbody>
      </table>
    </section>
  </main>
  <script id="graph-data" type="application/json">{graph_json}</script>
  <script>
  const graph = JSON.parse(document.getElementById("graph-data").textContent);
  const canvas = document.getElementById("kgCanvas");
  const ctx = canvas.getContext("2d");
  const searchBox = document.getElementById("searchBox");
  const graphMode = document.getElementById("graphMode");
  const toggleLabels = document.getElementById("toggleLabels");
  const fitGraph = document.getElementById("fitGraph");
  const maxNodes = document.getElementById("maxNodes");
  const maxNodesValue = document.getElementById("maxNodesValue");
  const legend = document.getElementById("legend");
  const relationFilters = document.getElementById("relationFilters");
  const details = document.getElementById("details");

  const relationColors = {{
    "isA": "#94a3b8",
    "USES_METHOD": "#2563eb",
    "USES_MATERIAL": "#16a34a",
    "HAS_CONDITION": "#f59e0b",
    "MEASURES_METRIC": "#7c3aed",
    "REPORTS_RESULT": "#dc2626"
  }};
  const nodeColors = {{
    "root": "#111827",
    "category": "#2563eb",
    "term": "#e2e8f0",
    "Material": "#16a34a",
    "Component": "#0f766e",
    "Method": "#2563eb",
    "Experiment": "#db2777",
    "ExperimentalSetting": "#f59e0b",
    "Condition": "#f97316",
    "Metric": "#7c3aed",
    "Result": "#dc2626",
    "QuantityValue": "#9333ea"
  }};
  const selectedRelations = new Set(graph.edges.map(edge => edge.relation));
  let visibleNodes = [];
  let visibleEdges = [];
  let showLabels = true;
  let selectedItem = null;
  let draggedNode = null;
  let pan = {{ x: 0, y: 0 }};
  let scale = 1;
  let needsSim = true;

  function resizeCanvas() {{
    const rect = canvas.parentElement.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(900, rect.width) * dpr;
    canvas.height = 720 * dpr;
    canvas.style.height = "720px";
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }}

  function shortLabel(value, max = 34) {{
    value = String(value || "");
    return value.length <= max ? value : value.slice(0, max - 1) + "…";
  }}

  function relationColor(rel) {{
    return relationColors[rel] || "#64748b";
  }}

  function nodeColor(node) {{
    return nodeColors[node.type] || "#cbd5e1";
  }}

  function isRelationEdge(edge) {{
    return String(edge.provenance || "").startsWith("relation");
  }}

  function edgeAllowed(edge) {{
    const mode = graphMode.value;
    if (mode === "relation" && !isRelationEdge(edge)) return false;
    if (mode === "taxonomy" && edge.relation !== "isA") return false;
    if (!selectedRelations.has(edge.relation)) return false;
    const query = searchBox.value.trim().toLowerCase();
    if (!query) return true;
    return [edge.source, edge.target, edge.relation, edge.provenance, edge.evidence_quote]
      .some(v => String(v || "").toLowerCase().includes(query));
  }}

  function rebuildGraph() {{
    const edgeCandidates = graph.edges.filter(edgeAllowed);
    const degree = new Map();
    edgeCandidates.forEach(edge => {{
      degree.set(edge.source, (degree.get(edge.source) || 0) + 1);
      degree.set(edge.target, (degree.get(edge.target) || 0) + 1);
    }});
    const nodeById = new Map(graph.nodes.map(node => [node.id, {{ ...node }}]));
    const nodeIds = Array.from(degree.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, Number(maxNodes.value))
      .map(([id]) => id);
    const nodeSet = new Set(nodeIds);
    visibleEdges = edgeCandidates
      .filter(edge => nodeSet.has(edge.source) && nodeSet.has(edge.target))
      .slice(0, 1300);
    visibleNodes = nodeIds
      .map(id => nodeById.get(id))
      .filter(Boolean);
    const rect = canvas.getBoundingClientRect();
    visibleNodes.forEach((node, index) => {{
      if (typeof node.x !== "number" || needsSim) {{
        const angle = (index / Math.max(1, visibleNodes.length)) * Math.PI * 2;
        const radius = 80 + (index % 13) * 18;
        node.x = rect.width / 2 + Math.cos(angle) * radius;
        node.y = rect.height / 2 + Math.sin(angle) * radius;
        node.vx = 0;
        node.vy = 0;
      }}
    }});
    needsSim = true;
    runSimulation(180);
    selectedItem = null;
    renderDetails(null);
    draw();
  }}

  function runSimulation(iterations) {{
    const nodes = visibleNodes;
    const nodeById = new Map(nodes.map(node => [node.id, node]));
    const links = visibleEdges.map(edge => ({{
      ...edge,
      sourceNode: nodeById.get(edge.source),
      targetNode: nodeById.get(edge.target)
    }})).filter(edge => edge.sourceNode && edge.targetNode);
    const width = canvas.getBoundingClientRect().width;
    const height = canvas.getBoundingClientRect().height;
    for (let tick = 0; tick < iterations; tick++) {{
      for (let i = 0; i < nodes.length; i++) {{
        const a = nodes[i];
        for (let j = i + 1; j < nodes.length; j++) {{
          const b = nodes[j];
          const dx = a.x - b.x || 0.01;
          const dy = a.y - b.y || 0.01;
          const dist2 = dx * dx + dy * dy;
          const force = Math.min(1400 / dist2, 0.9);
          a.vx += dx * force;
          a.vy += dy * force;
          b.vx -= dx * force;
          b.vy -= dy * force;
        }}
      }}
      links.forEach(edge => {{
        const a = edge.sourceNode;
        const b = edge.targetNode;
        const dx = b.x - a.x;
        const dy = b.y - a.y;
        const dist = Math.sqrt(dx * dx + dy * dy) || 1;
        const target = edge.relation === "isA" ? 110 : 160;
        const force = (dist - target) * 0.015;
        const fx = (dx / dist) * force;
        const fy = (dy / dist) * force;
        a.vx += fx;
        a.vy += fy;
        b.vx -= fx;
        b.vy -= fy;
      }});
      nodes.forEach(node => {{
        node.vx += (width / 2 - node.x) * 0.002;
        node.vy += (height / 2 - node.y) * 0.002;
        node.vx *= 0.72;
        node.vy *= 0.72;
        node.x += node.vx;
        node.y += node.vy;
      }});
    }}
    needsSim = false;
  }}

  function worldToScreen(node) {{
    return {{ x: node.x * scale + pan.x, y: node.y * scale + pan.y }};
  }}

  function screenToWorld(x, y) {{
    return {{ x: (x - pan.x) / scale, y: (y - pan.y) / scale }};
  }}

  function drawArrow(from, to, color) {{
    const dx = to.x - from.x;
    const dy = to.y - from.y;
    const dist = Math.sqrt(dx * dx + dy * dy) || 1;
    const ux = dx / dist;
    const uy = dy / dist;
    const start = {{ x: from.x + ux * 13, y: from.y + uy * 13 }};
    const end = {{ x: to.x - ux * 13, y: to.y - uy * 13 }};
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.3;
    ctx.beginPath();
    ctx.moveTo(start.x, start.y);
    ctx.lineTo(end.x, end.y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(end.x, end.y);
    ctx.lineTo(end.x - ux * 10 - uy * 4, end.y - uy * 10 + ux * 4);
    ctx.lineTo(end.x - ux * 10 + uy * 4, end.y - uy * 10 - ux * 4);
    ctx.closePath();
    ctx.fillStyle = color;
    ctx.fill();
  }}

  function draw() {{
    const rect = canvas.getBoundingClientRect();
    ctx.clearRect(0, 0, rect.width, rect.height);
    const nodeById = new Map(visibleNodes.map(node => [node.id, node]));
    ctx.font = "11px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
    visibleEdges.forEach(edge => {{
      const a = nodeById.get(edge.source);
      const b = nodeById.get(edge.target);
      if (!a || !b) return;
      const from = worldToScreen(a);
      const to = worldToScreen(b);
      const color = relationColor(edge.relation);
      ctx.globalAlpha = edge === selectedItem ? 1 : 0.55;
      drawArrow(from, to, color);
      if (showLabels && (edge.relation !== "isA" || graphMode.value !== "relation")) {{
        const mx = (from.x + to.x) / 2;
        const my = (from.y + to.y) / 2;
        const label = shortLabel(edge.relation, 20);
        const w = ctx.measureText(label).width + 10;
        ctx.globalAlpha = 0.92;
        ctx.fillStyle = "#ffffff";
        ctx.strokeStyle = color;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.roundRect(mx - w / 2, my - 9, w, 18, 5);
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = color;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(label, mx, my);
      }}
    }});
    ctx.globalAlpha = 1;
    visibleNodes.forEach(node => {{
      const p = worldToScreen(node);
      const color = nodeColor(node);
      const selected = node === selectedItem;
      ctx.fillStyle = color;
      ctx.strokeStyle = selected ? "#111827" : "#ffffff";
      ctx.lineWidth = selected ? 3 : 1.5;
      ctx.beginPath();
      ctx.arc(p.x, p.y, selected ? 11 : 8, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      if (selected || node.type === "category" || node.type === "root" || scale > 0.85) {{
        ctx.fillStyle = "#0f172a";
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.font = selected ? "600 12px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif" : "11px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
        ctx.fillText(shortLabel(node.label, selected ? 46 : 28), p.x + 11, p.y);
      }}
    }});
  }}

  function findNodeAt(x, y) {{
    for (let i = visibleNodes.length - 1; i >= 0; i--) {{
      const p = worldToScreen(visibleNodes[i]);
      const dx = x - p.x;
      const dy = y - p.y;
      if (dx * dx + dy * dy < 144) return visibleNodes[i];
    }}
    return null;
  }}

  function distanceToSegment(px, py, ax, ay, bx, by) {{
    const dx = bx - ax;
    const dy = by - ay;
    const len2 = dx * dx + dy * dy || 1;
    const t = Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / len2));
    const x = ax + t * dx;
    const y = ay + t * dy;
    return Math.hypot(px - x, py - y);
  }}

  function findEdgeAt(x, y) {{
    const nodeById = new Map(visibleNodes.map(node => [node.id, node]));
    for (let i = visibleEdges.length - 1; i >= 0; i--) {{
      const edge = visibleEdges[i];
      const a = nodeById.get(edge.source);
      const b = nodeById.get(edge.target);
      if (!a || !b) continue;
      const from = worldToScreen(a);
      const to = worldToScreen(b);
      if (distanceToSegment(x, y, from.x, from.y, to.x, to.y) < 7) return edge;
    }}
    return null;
  }}

  function renderDetails(item) {{
    if (!item) {{
      details.innerHTML = "Click a node or edge to inspect it.";
      return;
    }}
    if (item.source && item.target) {{
      details.innerHTML = `
        <h3>Edge</h3>
        <div class="kv"><strong>Relation</strong>${{escapeHtml(item.relation)}}</div>
        <div class="kv"><strong>Source</strong>${{escapeHtml(item.source)}}</div>
        <div class="kv"><strong>Target</strong>${{escapeHtml(item.target)}}</div>
        <div class="kv"><strong>Provenance</strong>${{escapeHtml(item.provenance || "")}}</div>
        <div class="kv"><strong>Confidence</strong>${{escapeHtml(item.confidence ?? "")}}</div>
        <div class="kv"><strong>Evidence</strong>${{escapeHtml(item.evidence_quote || "")}}</div>
      `;
    }} else {{
      const degree = visibleEdges.filter(edge => edge.source === item.id || edge.target === item.id).length;
      details.innerHTML = `
        <h3>Node</h3>
        <div class="kv"><strong>Label</strong>${{escapeHtml(item.label || item.id)}}</div>
        <div class="kv"><strong>Type</strong>${{escapeHtml(item.type || "")}}</div>
        <div class="kv"><strong>Visible degree</strong>${{degree}}</div>
      `;
    }}
  }}

  function escapeHtml(value) {{
    return String(value || "").replace(/[&<>"']/g, ch => ({{ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }}[ch]));
  }}

  function buildLegend() {{
    legend.innerHTML = Object.entries(relationColors).map(([rel, color]) =>
      `<div class="legend-item"><span class="swatch" style="background:${{color}}"></span><span>${{escapeHtml(rel)}}</span></div>`
    ).join("");
    const relations = Array.from(new Set(graph.edges.map(edge => edge.relation))).sort();
    relationFilters.innerHTML = relations.map(rel =>
      `<label><input type="checkbox" data-rel="${{escapeHtml(rel)}}" checked> ${{escapeHtml(rel)}}</label>`
    ).join("");
    relationFilters.addEventListener("change", event => {{
      const input = event.target;
      if (!input.matches("input[data-rel]")) return;
      if (input.checked) selectedRelations.add(input.dataset.rel);
      else selectedRelations.delete(input.dataset.rel);
      rebuildGraph();
    }});
  }}

  let lastMouse = null;
  canvas.addEventListener("mousedown", event => {{
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const node = findNodeAt(x, y);
    if (node) {{
      draggedNode = node;
      selectedItem = node;
      renderDetails(node);
      draw();
    }} else {{
      selectedItem = findEdgeAt(x, y);
      renderDetails(selectedItem);
      draw();
    }}
    lastMouse = {{ x, y }};
  }});
  canvas.addEventListener("mousemove", event => {{
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    if (draggedNode) {{
      const world = screenToWorld(x, y);
      draggedNode.x = world.x;
      draggedNode.y = world.y;
      draggedNode.vx = 0;
      draggedNode.vy = 0;
      draw();
    }} else if (event.buttons === 1 && lastMouse) {{
      pan.x += x - lastMouse.x;
      pan.y += y - lastMouse.y;
      draw();
    }}
    lastMouse = {{ x, y }};
  }});
  window.addEventListener("mouseup", () => {{ draggedNode = null; lastMouse = null; }});
  canvas.addEventListener("wheel", event => {{
    event.preventDefault();
    const delta = event.deltaY > 0 ? 0.9 : 1.1;
    scale = Math.max(0.25, Math.min(2.4, scale * delta));
    draw();
  }}, {{ passive: false }});

  searchBox.addEventListener("input", rebuildGraph);
  graphMode.addEventListener("change", rebuildGraph);
  maxNodes.addEventListener("input", () => {{
    maxNodesValue.textContent = maxNodes.value;
    rebuildGraph();
  }});
  toggleLabels.addEventListener("click", () => {{
    showLabels = !showLabels;
    toggleLabels.classList.toggle("active", showLabels);
    draw();
  }});
  fitGraph.addEventListener("click", () => {{
    pan = {{ x: 0, y: 0 }};
    scale = 1;
    rebuildGraph();
  }});
  window.addEventListener("resize", () => {{ resizeCanvas(); draw(); }});

  resizeCanvas();
  buildLegend();
  rebuildGraph();
  </script>
</body>
</html>
"""


NODE_TYPE_COLORS = {
    "root": "#111827",
    "category": "#2563eb",
    "term": "#cbd5e1",
    "Material": "#ff6b6b",
    "Component": "#0f766e",
    "Method": "#45b7d1",
    "Experiment": "#db2777",
    "ExperimentalSetting": "#f59e0b",
    "Condition": "#f97316",
    "Metric": "#7c3aed",
    "Result": "#dc2626",
    "QuantityValue": "#9333ea",
    "relation_entity": "#94a3b8",
}

RELATION_COLOR_HINTS = {
    "isA": "#94a3b8",
    "USES_METHOD": "#45b7d1",
    "USES_MATERIAL": "#16a34a",
    "HAS_CONDITION": "#f59e0b",
    "MEASURES_METRIC": "#7c3aed",
    "REPORTS_RESULT": "#dc2626",
}

FALLBACK_RELATION_COLORS = [
    "#2563eb",
    "#16a34a",
    "#f59e0b",
    "#7c3aed",
    "#dc2626",
    "#db2777",
    "#0f766e",
    "#475569",
]


def _stable_palette_color(value: str, palette: list[str]) -> str:
    if not value:
        return palette[0]
    return palette[sum(ord(char) for char in value) % len(palette)]


def _node_type_color(node_type: str | None) -> str:
    if node_type in NODE_TYPE_COLORS:
        return NODE_TYPE_COLORS[str(node_type)]
    return "#94a3b8"


def _relation_color(relation: str | None) -> str:
    relation_key = str(relation or "")
    if relation_key in RELATION_COLOR_HINTS:
        return RELATION_COLOR_HINTS[relation_key]
    return _stable_palette_color(relation_key, FALLBACK_RELATION_COLORS)


def _degree_by_node(edges: list[dict[str, Any]]) -> Counter[str]:
    degree: Counter[str] = Counter()
    for edge in edges:
        source = str(edge.get("source", "")).strip()
        target = str(edge.get("target", "")).strip()
        if source:
            degree[source] += 1
        if target:
            degree[target] += 1
    return degree


def _dedupe_nodes(graph: dict[str, Any]) -> list[dict[str, Any]]:
    nodes_by_id: dict[str, dict[str, Any]] = {}
    for node in graph.get("nodes", []):
        node_id = str(node.get("id", "")).strip()
        if not node_id or node_id in nodes_by_id:
            continue
        nodes_by_id[node_id] = {
            "id": node_id,
            "label": str(node.get("label") or node_id),
            "type": node.get("type") or "unknown",
            "level": node.get("level"),
        }

    for edge in graph.get("edges", []):
        for key in ("source", "target"):
            node_id = str(edge.get(key, "")).strip()
            if node_id and node_id not in nodes_by_id:
                nodes_by_id[node_id] = {
                    "id": node_id,
                    "label": node_id,
                    "type": "unknown",
                    "level": None,
                }
    return list(nodes_by_id.values())


def _vis_title(rows: list[tuple[str, Any]]) -> str:
    parts = []
    for key, value in rows:
        if value is None or value == "":
            continue
        parts.append(f"<strong>{html.escape(key)}</strong>: {html.escape(str(value))}")
    return "<br>".join(parts)


def _node_size(degree: int, max_degree: int, node_type: str | None) -> int:
    if node_type == "root":
        return 32
    if node_type == "category":
        return 24
    if max_degree <= 0:
        return 12
    scaled = 12 + int((degree / max_degree) * 23)
    return max(12, min(35, scaled))


def _to_vis_nodes(
    graph: dict[str, Any],
    max_label_chars: int,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    nodes = _dedupe_nodes(graph)
    degree = _degree_by_node(graph.get("edges", []))
    max_degree = max(degree.values(), default=0)
    node_type_colors: dict[str, str] = {}
    vis_nodes = []

    for node in nodes:
        node_id = str(node["id"])
        node_type = str(node.get("type") or "unknown")
        node_degree = degree.get(node_id, 0)
        color = _node_type_color(node_type)
        node_type_colors[node_type] = color
        vis_nodes.append(
            {
                "id": node_id,
                "label": _short_label(str(node.get("label") or node_id), max_label_chars),
                "title": _vis_title(
                    [
                        ("Node", node.get("label") or node_id),
                        ("Type", node_type),
                        ("Degree", node_degree),
                    ]
                ),
                "shape": "dot",
                "size": _node_size(node_degree, max_degree, node_type),
                "color": {
                    "background": color,
                    "border": "#0f172a" if node_type in {"root", "category"} else color,
                    "highlight": {
                        "background": color,
                        "border": "#111827",
                    },
                },
                "font": {
                    "size": 16 if node_type in {"root", "category"} else 13,
                    "color": "#111827",
                    "face": "-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif",
                },
                "node_type": node_type,
                "raw_label": str(node.get("label") or node_id),
                "degree": node_degree,
            }
        )
    return vis_nodes, node_type_colors


def _to_vis_edges(graph: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    relation_colors: dict[str, str] = {}
    vis_edges = []
    for index, edge in enumerate(graph.get("edges", [])):
        source = str(edge.get("source", "")).strip()
        target = str(edge.get("target", "")).strip()
        relation = str(edge.get("relation", "")).strip()
        if not source or not target or not relation:
            continue

        color = _relation_color(relation)
        relation_colors[relation] = color
        evidence = _short_label(str(edge.get("evidence_quote") or ""), 320)
        provenance = str(edge.get("provenance") or "")
        is_taxonomy = relation == "isA"
        is_refined = provenance.startswith("relation_refinement")
        vis_edges.append(
            {
                "id": f"edge-{index}",
                "from": source,
                "to": target,
                "source": source,
                "target": target,
                "label": relation,
                "relation": relation,
                "title": _vis_title(
                    [
                        ("Relation", relation),
                        ("Source", source),
                        ("Target", target),
                        ("Provenance", provenance),
                        ("Confidence", edge.get("confidence")),
                        ("Evidence", evidence),
                    ]
                ),
                "arrows": "to",
                "color": {
                    "color": color,
                    "highlight": "#111827",
                    "hover": color,
                    "opacity": 0.26 if is_taxonomy else 0.52,
                },
                "font": {
                    "size": 8 if is_taxonomy else 10,
                    "align": "middle",
                    "background": "rgba(255, 255, 255, 0.88)",
                    "strokeWidth": 0,
                    "color": color,
                },
                "width": 0.8 if is_taxonomy else 1.6,
                "smooth": False,
                "dashes": [6, 4] if is_refined else False,
                "provenance": provenance,
                "confidence": edge.get("confidence"),
                "evidence_quote": str(edge.get("evidence_quote") or ""),
            }
        )
    return vis_edges, relation_colors


def _render_vis_relation_rows(relation_edges: list[dict[str, Any]]) -> str:
    rows = []
    for edge in relation_edges:
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(edge.get('source', '')))}</td>"
            f"<td><strong>{html.escape(str(edge.get('relation', '')))}</strong></td>"
            f"<td>{html.escape(str(edge.get('target', '')))}</td>"
            f"<td>{html.escape(str(edge.get('confidence', '') or ''))}</td>"
            f"<td>{html.escape(str(edge.get('provenance', '')))}</td>"
            f"<td>{html.escape(_short_label(str(edge.get('evidence_quote', '') or ''), 240))}</td>"
            "</tr>"
        )
    if not rows:
        return '<tr><td colspan="6">No relation graph was found.</td></tr>'
    return "\n".join(rows)


def _render_vis_html(
    graph: dict[str, Any],
    taxonomy_root: LayoutNode,
    output_json: Path,
    max_label_chars: int,
) -> str:
    svg, width, height = _render_svg(taxonomy_root, max_label_chars)
    vis_nodes, node_type_colors = _to_vis_nodes(graph, max_label_chars)
    vis_edges, relation_colors = _to_vis_edges(graph)
    relation_edges = [
        edge for edge in graph["edges"] if str(edge.get("provenance", "")).startswith("relation")
    ]
    relation_counts = Counter(edge.get("relation") for edge in graph["edges"])
    provenance_counts = Counter(edge.get("provenance") for edge in graph["edges"])
    node_type_counts = Counter(node.get("type") for node in _dedupe_nodes(graph))
    payload = {
        "nodes": vis_nodes,
        "edges": vis_edges,
        "relations": sorted(relation_colors),
        "relationColors": relation_colors,
        "nodeTypeColors": node_type_colors,
    }
    template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OntoloGYM KG Visualization</title>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/dist/vis-network.min.css">
  <script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/vis-network.min.js"></script>
  <style>
    :root {
      --bg: #f6f7fb;
      --panel: #ffffff;
      --ink: #111827;
      --muted: #64748b;
      --line: #d8dee9;
      --accent: #2563eb;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--bg);
    }
    header {
      padding: 22px 28px 14px;
      background: var(--panel);
      border-bottom: 1px solid var(--line);
      position: sticky;
      top: 0;
      z-index: 10;
    }
    h1 { margin: 0 0 8px; font-size: 24px; letter-spacing: 0; }
    h2 { margin: 0 0 12px; font-size: 17px; letter-spacing: 0; }
    h3 { margin: 0 0 8px; font-size: 14px; letter-spacing: 0; }
    main { padding: 20px 28px 36px; display: grid; gap: 18px; }
    .meta { color: var(--muted); font-size: 14px; line-height: 1.45; }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      overflow: hidden;
    }
    .stats { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 6px 9px;
      border: 1px solid #dbe3ef;
      border-radius: 999px;
      background: #f8fafc;
      color: #334155;
      font-size: 12px;
    }
    .pill strong { color: #0f172a; }
    .workspace {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 360px;
      gap: 14px;
      min-height: 780px;
    }
    .network-panel {
      position: relative;
      min-width: 0;
      border: 1px solid #dbe3ef;
      border-radius: 8px;
      background: #ffffff;
      overflow: hidden;
    }
    .controls {
      position: absolute;
      left: 12px;
      top: 12px;
      right: 12px;
      z-index: 3;
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 8px;
      pointer-events: none;
    }
    .controls > * { pointer-events: auto; }
    input, select, button {
      height: 34px;
      border: 1px solid #cbd5e1;
      border-radius: 6px;
      background: #ffffff;
      color: #0f172a;
      font-size: 13px;
      padding: 0 10px;
    }
    button { cursor: pointer; font-weight: 600; }
    button:hover { border-color: var(--accent); color: var(--accent); }
    #mynetwork {
      width: 100%;
      height: 780px;
      background-color: #ffffff;
      position: relative;
    }
    #dependencyWarning {
      display: none;
      position: absolute;
      inset: 56px 18px auto;
      z-index: 5;
      padding: 12px;
      border: 1px solid #fecaca;
      border-radius: 8px;
      background: #fff1f2;
      color: #991b1b;
      font-size: 13px;
    }
    #loadingBar {
      position: absolute;
      inset: 0;
      height: 780px;
      background-color: rgba(248, 250, 252, 0.84);
      transition: opacity 0.5s ease;
      opacity: 1;
      z-index: 2;
    }
    .outerBorder {
      position: relative;
      top: 385px;
      width: 600px;
      max-width: calc(100% - 48px);
      height: 44px;
      margin: auto;
      border: 8px solid rgba(15, 23, 42, 0.08);
      background: #ffffff;
      border-radius: 72px;
      box-shadow: 0 0 10px rgba(15, 23, 42, 0.12);
    }
    #border {
      position: absolute;
      top: 10px;
      left: 10px;
      right: 84px;
      height: 23px;
      border-radius: 10px;
      box-shadow: inset 0 0 4px rgba(15, 23, 42, 0.18);
      overflow: hidden;
      background: #e2e8f0;
    }
    #bar {
      width: 20px;
      height: 23px;
      border-radius: 10px;
      background: #2563eb;
      box-shadow: 2px 0 4px rgba(15, 23, 42, 0.26);
    }
    #text {
      position: absolute;
      top: 6px;
      right: 18px;
      font-size: 18px;
      color: #0f172a;
      font-weight: 700;
    }
    .side {
      border: 1px solid #dbe3ef;
      border-radius: 8px;
      background: #ffffff;
      overflow: hidden;
      display: grid;
      grid-template-rows: auto 1fr;
      min-height: 780px;
    }
    .side-head { padding: 14px; border-bottom: 1px solid #e2e8f0; }
    .side-body { padding: 14px; overflow: auto; }
    .legend { display: grid; gap: 7px; font-size: 13px; }
    .legend-item { display: flex; align-items: center; gap: 8px; }
    .swatch {
      width: 14px;
      height: 14px;
      border-radius: 4px;
      border: 1px solid rgba(15, 23, 42, 0.15);
      flex: none;
    }
    .details {
      margin-top: 16px;
      padding-top: 14px;
      border-top: 1px solid #e2e8f0;
      font-size: 13px;
      line-height: 1.5;
    }
    .kv { margin: 7px 0; overflow-wrap: anywhere; }
    .kv strong {
      display: block;
      color: #475569;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    .relation-filter {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 10px;
    }
    .relation-filter label {
      display: inline-flex;
      gap: 5px;
      align-items: center;
      font-size: 12px;
      border: 1px solid #dbe3ef;
      border-radius: 999px;
      padding: 5px 8px;
      background: #f8fafc;
    }
    .relation-filter input { height: auto; }
    .taxonomy-panel { max-height: 520px; overflow: auto; }
    svg { min-width: __SVG_WIDTH__px; min-height: __SVG_HEIGHT__px; }
    .edge { fill: none; stroke: #94a3b8; stroke-width: 1.4; }
    text { font-size: 12px; dominant-baseline: middle; }
    .table-panel { max-height: 520px; overflow: auto; }
    table { border-collapse: collapse; width: 100%; font-size: 12px; }
    th, td {
      border-bottom: 1px solid #e2e8f0;
      padding: 8px 10px;
      text-align: left;
      vertical-align: top;
    }
    th { background: #f1f5f9; position: sticky; top: 0; z-index: 1; }
    td:nth-child(1), td:nth-child(3), td:nth-child(6) { max-width: 360px; overflow-wrap: anywhere; }
    @media (max-width: 1100px) {
      .workspace { grid-template-columns: 1fr; }
      .side { min-height: 420px; }
    }
  </style>
</head>
<body>
  <header>
    <h1>OntoloGYM KG Visualization</h1>
    <div class="meta">Nodes: __NODE_COUNT__ · Edges: __EDGE_COUNT__ · JSON: __OUTPUT_JSON__</div>
    <div class="stats">__RELATION_PILLS__</div>
  </header>
  <main>
    <section class="panel">
      <h2>Interactive Knowledge Graph</h2>
      <div class="meta">Built with vis-network. Edge labels show KG relations; use filters to switch between relation augmentation, isA taxonomy, and the full graph.</div>
      <div class="workspace">
        <div class="network-panel">
          <div class="controls">
            <input id="searchBox" placeholder="Search node, relation, evidence">
            <select id="graphMode" title="Graph mode">
              <option value="relation">relation edges</option>
              <option value="taxonomy">isA taxonomy</option>
              <option value="all">all edges</option>
            </select>
            <button id="fitGraph" type="button">fit</button>
            <button id="stabilizeGraph" type="button">stabilize</button>
            <button id="clearSelection" type="button">clear</button>
          </div>
          <div id="dependencyWarning">vis-network could not be loaded. Check the CDN connection, or bundle vis-network locally for offline use.</div>
          <div id="mynetwork"></div>
          <div id="loadingBar">
            <div class="outerBorder">
              <div id="text">0%</div>
              <div id="border"><div id="bar"></div></div>
            </div>
          </div>
        </div>
        <aside class="side">
          <div class="side-head">
            <h2>Legend & Details</h2>
            <div id="visibleStats" class="meta">Preparing graph...</div>
          </div>
          <div class="side-body">
            <h3>Relations</h3>
            <div id="legend" class="legend"></div>
            <div id="relationChecks" class="relation-filter"></div>
            <div id="details" class="details">Click a node or edge to inspect it.</div>
            <div class="details">
              <h3>Node Types</h3>
              <div class="stats">__NODE_TYPE_PILLS__</div>
            </div>
            <div class="details">
              <h3>Provenance</h3>
              <div class="stats">__PROVENANCE_PILLS__</div>
            </div>
          </div>
        </aside>
      </div>
    </section>
    <section class="panel">
      <h2>Ontology Taxonomy Tree</h2>
      <div class="meta">Static isA hierarchy from OntoGen. Use the interactive graph above for relation labels and evidence.</div>
      <div class="taxonomy-panel">__TAXONOMY_SVG__</div>
    </section>
    <section class="panel table-panel">
      <h2>Relation Edge Table</h2>
      <table>
        <thead><tr><th>Source</th><th>Relation</th><th>Target</th><th>Confidence</th><th>Provenance</th><th>Evidence</th></tr></thead>
        <tbody>__RELATION_ROWS__</tbody>
      </table>
    </section>
  </main>
  <script id="vis-data" type="application/json">__VIS_PAYLOAD__</script>
  <script>
  (function () {
    const payload = JSON.parse(document.getElementById("vis-data").textContent);
    const dependencyWarning = document.getElementById("dependencyWarning");
    const loadingBar = document.getElementById("loadingBar");
    const bar = document.getElementById("bar");
    const loadingText = document.getElementById("text");
    const searchBox = document.getElementById("searchBox");
    const graphMode = document.getElementById("graphMode");
    const fitGraph = document.getElementById("fitGraph");
    const stabilizeGraph = document.getElementById("stabilizeGraph");
    const clearSelection = document.getElementById("clearSelection");
    const visibleStats = document.getElementById("visibleStats");
    const legend = document.getElementById("legend");
    const relationChecks = document.getElementById("relationChecks");
    const details = document.getElementById("details");

    if (!window.vis) {
      dependencyWarning.style.display = "block";
      loadingBar.style.display = "none";
      return;
    }

    const allNodes = payload.nodes;
    const allEdges = payload.edges;
    const nodeDataById = new Map(allNodes.map(node => [node.id, node]));
    const edgeDataById = new Map(allEdges.map(edge => [edge.id, edge]));
    const selectedRelations = new Set(payload.relations);
    const nodes = new vis.DataSet([]);
    const edges = new vis.DataSet([]);
    const container = document.getElementById("mynetwork");
    const data = { nodes, edges };
    const options = {
      nodes: {
        shape: "dot",
        borderWidth: 1.5,
        scaling: { min: 10, max: 35 },
        font: {
          size: 14,
          face: "-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif",
          color: "#111827",
        },
      },
      edges: {
        arrows: { to: { enabled: true, scaleFactor: 0.68 } },
        font: {
          size: 9,
          align: "middle",
          background: "rgba(255, 255, 255, 0.88)",
          strokeWidth: 0,
        },
        color: { opacity: 0.45 },
        smooth: false,
        width: 1.2,
      },
      physics: {
        enabled: true,
        stabilization: { enabled: true, iterations: 300, updateInterval: 25 },
        barnesHut: {
          gravitationalConstant: -7200,
          centralGravity: 0.18,
          springLength: 135,
          springConstant: 0.035,
          damping: 0.24,
          avoidOverlap: 0.08,
        },
      },
      interaction: {
        hover: true,
        tooltipDelay: 100,
        hideEdgesOnDrag: true,
        hideNodesOnDrag: false,
        multiselect: true,
        navigationButtons: true,
        keyboard: true,
      },
    };
    const network = new vis.Network(container, data, options);
    let loadingFallbackTimer = null;

    function showLoading() {
      window.clearTimeout(loadingFallbackTimer);
      loadingBar.style.display = "block";
      loadingBar.style.opacity = 1;
      bar.style.width = "20px";
      loadingText.textContent = "0%";
    }

    function hideLoading(shouldFit) {
      window.clearTimeout(loadingFallbackTimer);
      loadingText.textContent = "100%";
      bar.style.width = "100%";
      loadingBar.style.opacity = 0;
      window.setTimeout(function () {
        loadingBar.style.display = "none";
      }, 300);
      if (shouldFit !== false) {
        network.fit({ animation: { duration: 350, easingFunction: "easeInOutQuad" } });
      }
    }

    function hideLoadingAfterTimeout() {
      window.clearTimeout(loadingFallbackTimer);
      loadingFallbackTimer = window.setTimeout(function () {
        hideLoading(true);
      }, 2500);
    }

    function escapeHtml(value) {
      return String(value || "").replace(/[&<>"']/g, function (char) {
        return ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[char];
      });
    }

    function isRelationEdge(edge) {
      return String(edge.provenance || "").indexOf("relation") === 0;
    }

    function edgeModeMatches(edge) {
      if (graphMode.value === "relation") return isRelationEdge(edge);
      if (graphMode.value === "taxonomy") return edge.relation === "isA";
      return true;
    }

    function searchableEdgeText(edge) {
      return [
        edge.from,
        edge.to,
        edge.relation,
        edge.provenance,
        edge.confidence,
        edge.evidence_quote,
      ].join(" ").toLowerCase();
    }

    function searchableNodeText(node) {
      return [node.raw_label, node.label, node.node_type].join(" ").toLowerCase();
    }

    function currentEdges() {
      const query = searchBox.value.trim().toLowerCase();
      return allEdges.filter(edge => {
        if (!edgeModeMatches(edge)) return false;
        if (!selectedRelations.has(edge.relation)) return false;
        if (!query) return true;
        return searchableEdgeText(edge).includes(query);
      });
    }

    function applyFilters(shouldFit) {
      const filteredEdges = currentEdges();
      const nodeIds = new Set();
      filteredEdges.forEach(edge => {
        nodeIds.add(edge.from);
        nodeIds.add(edge.to);
      });

      const query = searchBox.value.trim().toLowerCase();
      if (query) {
        allNodes.forEach(node => {
          if (searchableNodeText(node).includes(query)) nodeIds.add(node.id);
        });
      }

      const filteredNodes = allNodes.filter(node => nodeIds.has(node.id));
      const visibleNodeIds = new Set(filteredNodes.map(node => node.id));
      const safeEdges = filteredEdges.filter(edge => visibleNodeIds.has(edge.from) && visibleNodeIds.has(edge.to));

      nodes.clear();
      edges.clear();
      nodes.add(filteredNodes);
      edges.add(safeEdges);
      visibleStats.textContent = `${filteredNodes.length} visible nodes · ${safeEdges.length} visible edges`;

      if (!filteredNodes.length) {
        hideLoading(false);
        return;
      }

      if (shouldFit !== false) {
        showLoading();
        network.stabilize(180);
        hideLoadingAfterTimeout();
      }
    }

    function buildLegend() {
      legend.innerHTML = "";
      Object.entries(payload.relationColors).forEach(([relation, color]) => {
        const item = document.createElement("div");
        item.className = "legend-item";
        const swatch = document.createElement("span");
        swatch.className = "swatch";
        swatch.style.background = color;
        const label = document.createElement("span");
        label.textContent = relation;
        item.append(swatch, label);
        legend.append(item);
      });

      payload.relations.forEach(relation => {
        const label = document.createElement("label");
        const input = document.createElement("input");
        input.type = "checkbox";
        input.checked = true;
        input.dataset.relation = relation;
        const span = document.createElement("span");
        span.textContent = relation;
        label.append(input, span);
        relationChecks.append(label);
      });
    }

    function renderNodeDetails(nodeId) {
      const node = nodeDataById.get(nodeId);
      if (!node) return;
      const connected = allEdges
        .filter(edge => edge.from === nodeId || edge.to === nodeId)
        .slice(0, 12)
        .map(edge => `<div class="kv">${escapeHtml(edge.from)} <strong>${escapeHtml(edge.relation)}</strong> ${escapeHtml(edge.to)}</div>`)
        .join("");
      details.innerHTML = `
        <h3>Node</h3>
        <div class="kv"><strong>Label</strong>${escapeHtml(node.raw_label || node.label)}</div>
        <div class="kv"><strong>Type</strong>${escapeHtml(node.node_type)}</div>
        <div class="kv"><strong>Total degree</strong>${escapeHtml(node.degree)}</div>
        <div class="kv"><strong>Connected edges</strong>${connected || "No connected edges in current graph."}</div>
      `;
    }

    function renderEdgeDetails(edgeId) {
      const edge = edgeDataById.get(edgeId);
      if (!edge) return;
      details.innerHTML = `
        <h3>Edge</h3>
        <div class="kv"><strong>Relation</strong>${escapeHtml(edge.relation)}</div>
        <div class="kv"><strong>Source</strong>${escapeHtml(edge.from)}</div>
        <div class="kv"><strong>Target</strong>${escapeHtml(edge.to)}</div>
        <div class="kv"><strong>Provenance</strong>${escapeHtml(edge.provenance)}</div>
        <div class="kv"><strong>Confidence</strong>${escapeHtml(edge.confidence ?? "")}</div>
        <div class="kv"><strong>Evidence</strong>${escapeHtml(edge.evidence_quote || "")}</div>
      `;
    }

    network.on("stabilizationProgress", function (params) {
      loadingBar.style.display = "block";
      loadingBar.style.opacity = 1;
      const maxWidth = Math.max(20, document.getElementById("border").clientWidth);
      const total = Math.max(1, Number(params.total) || Number(params.iterations) || 1);
      const widthFactor = Math.max(0, Math.min(1, Number(params.iterations) / total));
      const width = Math.max(20, maxWidth * widthFactor);
      bar.style.width = width + "px";
      loadingText.textContent = Math.round(widthFactor * 100) + "%";
    });

    network.on("stabilizationIterationsDone", function () {
      hideLoading(true);
    });

    network.on("stabilized", function () {
      hideLoading(false);
    });

    network.on("click", function (params) {
      if (params.nodes.length) {
        renderNodeDetails(params.nodes[0]);
        return;
      }
      if (params.edges.length) {
        renderEdgeDetails(params.edges[0]);
        return;
      }
      details.textContent = "Click a node or edge to inspect it.";
    });

    network.on("doubleClick", function (params) {
      if (params.nodes.length) {
        network.focus(params.nodes[0], {
          scale: 1.1,
          animation: { duration: 300, easingFunction: "easeInOutQuad" },
        });
      }
    });

    relationChecks.addEventListener("change", function (event) {
      const input = event.target;
      if (!input.matches("input[data-relation]")) return;
      if (input.checked) selectedRelations.add(input.dataset.relation);
      else selectedRelations.delete(input.dataset.relation);
      applyFilters(true);
    });
    searchBox.addEventListener("input", function () { applyFilters(true); });
    graphMode.addEventListener("change", function () { applyFilters(true); });
    fitGraph.addEventListener("click", function () {
      network.fit({ animation: { duration: 300, easingFunction: "easeInOutQuad" } });
    });
    stabilizeGraph.addEventListener("click", function () {
      showLoading();
      network.stabilize(180);
      hideLoadingAfterTimeout();
    });
    clearSelection.addEventListener("click", function () {
      network.unselectAll();
      details.textContent = "Click a node or edge to inspect it.";
    });

    buildLegend();
    applyFilters(true);
  })();
  </script>
</body>
</html>
"""
    return (
        template.replace("__SVG_WIDTH__", str(width))
        .replace("__SVG_HEIGHT__", str(height))
        .replace("__NODE_COUNT__", str(len(vis_nodes)))
        .replace("__EDGE_COUNT__", str(len(vis_edges)))
        .replace("__OUTPUT_JSON__", html.escape(str(output_json)))
        .replace("__RELATION_PILLS__", _counter_summary(relation_counts, len(vis_edges)))
        .replace("__NODE_TYPE_PILLS__", _counter_summary(node_type_counts, len(vis_nodes)))
        .replace("__PROVENANCE_PILLS__", _counter_summary(provenance_counts, len(vis_edges)))
        .replace("__TAXONOMY_SVG__", svg)
        .replace("__RELATION_ROWS__", _render_vis_relation_rows(relation_edges))
        .replace("__VIS_PAYLOAD__", _json_for_script(payload))
    )


def run_visualization() -> dict[str, Any]:
    config = load_project_config()
    taxonomy_path = Path(config.KG_VIS_TAXONOMY_PICKLE)
    output_json = Path(config.KG_VIS_GRAPH_JSON)
    output_html = Path(config.KG_VIS_HTML)
    output_html.parent.mkdir(parents=True, exist_ok=True)

    taxonomy_tree = _load_taxonomy_tree(taxonomy_path)
    taxonomy_root = _tree_to_layout_node(taxonomy_tree)
    nodes, edges = _collect_graph(taxonomy_root)
    relation_sources = [Path(config.KG_VIS_RELATION_GRAPH_JSON)]
    relation_sources.extend(Path(path) for path in getattr(config, "KG_VIS_EXTRA_RELATION_GRAPH_JSONS", []))
    for index, relation_path in enumerate(relation_sources):
        provenance = "relation_augmentation" if index == 0 else f"relation_refinement_{index}"
        _merge_relation_graph(nodes, edges, relation_path, provenance)
    graph = {
        "nodes": nodes,
        "edges": edges,
        "sources": {
            "taxonomy_pickle": str(taxonomy_path),
            "relation_graph_json": str(config.KG_VIS_RELATION_GRAPH_JSON),
            "extra_relation_graph_jsons": [str(path) for path in relation_sources[1:]],
        },
    }

    output_json.write_text(json.dumps(graph, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_html.write_text(
        _render_vis_html(
            graph=graph,
            taxonomy_root=taxonomy_root,
            output_json=output_json,
            max_label_chars=int(getattr(config, "KG_VIS_MAX_LABEL_CHARS", 72)),
        ),
        encoding="utf-8",
    )

    result = {
        "nodes": len(nodes),
        "edges": len(edges),
        "json": str(output_json),
        "html": str(output_html),
    }
    record_pipeline_run(
        config.RUN_OUTPUT_DIR,
        "kg_visualization",
        status="completed",
        inputs={
            "taxonomy_pickle": str(taxonomy_path),
            "relation_graph_json": str(config.KG_VIS_RELATION_GRAPH_JSON),
        },
        outputs={
            "graph_json": str(output_json),
            "html": str(output_html),
        },
        extra={
            "nodes": len(nodes),
            "edges": len(edges),
        },
    )
    return result
