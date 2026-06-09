"""KG 산출물을 눈으로 확인하기 위한 시각화 설정."""

from configs.common import RUN_OUTPUT_DIR
from configs.kg_refinement import KG_REFINE_GRAPH_JSON
from configs.ontogen import ONTOGEN_TAXONOMY_DIR
from configs.relation_augmentation import RELATION_GRAPH_JSON


KG_VIS_OUTPUT_DIR = RUN_OUTPUT_DIR / "kg_visualization"
KG_VIS_GRAPH_JSON = KG_VIS_OUTPUT_DIR / "kg_graph.json"
KG_VIS_HTML = KG_VIS_OUTPUT_DIR / "kg_graph.html"

# OntoGen taxonomy pickle을 기본 KG로 사용합니다.
KG_VIS_TAXONOMY_PICKLE = ONTOGEN_TAXONOMY_DIR / "tree_0.pkl"

# relation augmentation 결과가 있으면 HTML 하단에 함께 표시합니다.
KG_VIS_RELATION_GRAPH_JSON = RELATION_GRAPH_JSON
KG_VIS_EXTRA_RELATION_GRAPH_JSONS = [KG_REFINE_GRAPH_JSON]

# 너무 긴 논문 문장형 node가 화면을 덮지 않도록 라벨을 줄입니다.
KG_VIS_MAX_LABEL_CHARS = 72
