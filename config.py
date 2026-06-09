"""
기존 코드 호환용 통합 설정 파일.

실제로 수정할 설정은 configs/ 폴더에 파이프라인별로 나뉘어 있습니다.
- configs/common.py
- configs/qa_extractor.py
- configs/ontogen.py
- configs/relation_augmentation.py
- configs/qa_evaluation.py
- configs/kg_refinement.py
- configs/kg_visualization.py
"""

from configs.common import *  # noqa: F401,F403
from configs.qa_extractor import *  # noqa: F401,F403
from configs.ontogen import *  # noqa: F401,F403
from configs.relation_augmentation import *  # noqa: F401,F403
from configs.qa_evaluation import *  # noqa: F401,F403
from configs.kg_refinement import *  # noqa: F401,F403
from configs.kg_visualization import *  # noqa: F401,F403
