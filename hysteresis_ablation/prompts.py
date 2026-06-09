"""Domain prompts used by the hysteresis-loop ablation code."""

HYSTERESIS_LOOP_CONTEXT = """
Hysteresis-loop context for permanent-magnet papers:

A magnetic hysteresis loop shows how a magnetic material responds to an applied magnetic field.
The x-axis is usually magnetic field H or B, and the y-axis is usually magnetization M,
magnetic flux density B, magnetic polarization J, resistance, or magnetoresistance when a
transport signal is used as the magnetic-state readout. In permanent-magnet research, the
important visual evidence is usually the loop width, height, rectangularity, second-quadrant
demagnetization behavior, and differences between curves, samples, temperatures, directions,
or processing conditions.

Key quantities and visual cues:
- Saturation magnetization Ms: high-field magnetization when the curve approaches saturation.
- Remanence Mr, Br, or Jr: remaining magnetization, flux density, or polarization at zero field.
- Coercivity Hc, Hcj, or Hcb: reverse field where magnetization, flux density, or polarization crosses zero.
- Squareness: how rectangular the loop is; higher squareness usually means stronger permanent-magnet behavior.
- Maximum energy product (BH)max: usually from a second-quadrant demagnetization curve when shown or captioned.
- Loop area: related to hysteresis loss; larger area means stronger hysteresis.
- Wider loops generally indicate higher coercivity, taller loops indicate higher magnetization or remanence,
  and more rectangular loops indicate better permanent-magnet characteristics.

Extraction rules for the current KG schema:
- Use the configured relation schema only. Do not invent relation or entity types.
- Extract numerical values only when visible in the figure or explicitly stated in the caption.
- Preserve units exactly, such as kOe, Oe, T, A/m, emu/g, emu/cm3, kA/m, or kJ/m3.
- Distinguish M-H, B-H, J-H, and transport hysteresis loops when the axes or caption make this clear.
- Treat each curve as representing a sample, composition, processing condition, temperature, or field direction
  only when the caption, legend, or visible label supports that mapping.
- Do not infer causality from the figure alone. Prefer visual observations such as "shows higher coercivity",
  "has a wider loop", or "has larger remanence" unless the caption explicitly says a method improved a property.
- If a value, curve identity, or condition is unclear, use UNKNOWN rather than guessing.
- For visual-only evidence that is not a literal caption quote, set evidence_quote to VISIBLE_IN_FIGURE and
  put a short explanation in qualifiers.visual_evidence.
"""


HYSTERESIS_FIGURE_CLASSIFIER_PROMPT = """
Classify whether the provided scientific figure is a magnetic hysteresis-loop figure.

Use the example concept of hysteresis loops: closed or loop-like curves measured against a swept magnetic
field, often labeled M-H, B-H, J-H, H, magnetic field, magnetization, coercivity, remanence, demagnetization,
FORC, MR vs field with sweep direction, or similar. Multi-panel figures count as positive if at least one
panel contains a hysteresis loop or demagnetization curve relevant to magnetic switching/permanent magnets.

Caption:
{caption}

Return a JSON object only:
{{
  "is_hysteresis_loop": true,
  "confidence": 0.0,
  "reason": "short reason grounded in caption or visible axes/curves"
}}
"""


HYSTERESIS_QA_FOCUS_HINT = """
Generate a question that requires inspecting the hysteresis-loop image together with its caption.
Prefer questions about axes, curve labels, sample or condition mapping, coercivity/remanence/saturation
comparisons, loop width, loop height, squareness, second-quadrant demagnetization behavior, or visually
reported numerical values. The answer must be supported by the image and caption. Do not ask a question
whose answer is only general magnetic-domain knowledge or only "not reported".
"""
