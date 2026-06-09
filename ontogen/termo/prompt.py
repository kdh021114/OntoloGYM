"""
A collection of prompt templates for extracting terms, acronyms, verbs, triplets, and definitions from scientific texts.
"""

prompt_abstract = """
Given this scientific text:

===
{text}
===

Extract compact ontology node labels from the scientific text. Keep reusable scientific entities, concepts, methods, materials, instruments, metrics, conditions, and result types. Do not include section titles, full sentences, figure/table references by themselves, author names, journal names, institutions, DOI, or publishing metadata. Prefer concise noun phrases of 2-6 words. Return only bullet lines beginning with "- ".
"""


# Given the following context and the following vocabulary, identify all the acronyms and symbols in the vocabulary and their corresponding terms, according to the context.
prompt_acronym = """
Given the following context and the following vocabulary, identify all the acronyms and symbols in the context and their corresponding terms in the vocabulary, according to the context. 
Do not print anything else other than the acronyms and their corresponding terms.
Provide the results in the format:

<acronym1>: <term1>
<acronym2>: <term2>

Context:

===
{CONTEXT}
===

Vocabulary:

===
{VOCABULARY}
===
"""


prompt_verbs = """
Given this scientific text:

===
{CONTEXT}
===

Extract every verb from the scientific text, including both single-word verbs and composed verbs. Ensure to capture all specific technical verbs. Do not include non-scientific verbs such as verbs related to authors, journals, institutions, editorial information or any publishing related data (e.g. 'published'). Output a list that starts with '-'.
"""


prompt_triplets = """
Given this scientific text:
===
{CONTEXT}
===
This vocabulary:
===
{VOCABULARY}
===
Extract every triplet (term > relationship > term) from the scientific text that involves the terms presented in the vocabulary. Extract only triplets that explicitely appear in the text. Use only the terms presented in the vocabulary. Focus only in short relationships, with around five or less words. Do not ouput long relationships. Output a list that starts with '-'.
"""

prompt_definitions = """
Given this scientific text:
===
{CONTEXT}
===
This vocabulary:
===
{VOCABULARY}
===
Extract the definition of each term in the vocabulary from the scientific text. For the definitions, use only the context provided, this is, do not include any informaiton in the definitions that is not present in the provided text. Focus only in short, brief and very concise definitions. Do not ouput long definitions. If no explicit definition is provided, provide the information available in the context to undestand the meaning of the term. Output a list that starts with '-'.
"""


prompt_evidence_terms = """
Given this evidence-aware scientific context:

===
{text}
===

Extract only compact ontology node labels for a knowledge graph.

Keep a term only if it is a reusable scientific entity, concept, method, material, instrument, metric, experimental condition, or result type. Good nodes are noun phrases such as:
- exchange bias field
- cooling field Hcool
- X-ray reflectivity
- FORC distribution
- CoO antiferromagnet
- soft magnet multilayer
- hysteresis loop measurement
- coercivity Hc

Do NOT output:
- section names or buckets such as Domain terms, Evidence terms, Table-derived labels, Experimental settings
- figure/table/equation references by themselves, e.g. Figure 3, Figure 4a, left panel, black arrows
- full sentences, clauses, claims, observations, or caption fragments
- phrases with verbs such as "is", "are", "was", "were", "shows", "indicates", "causes", "leads to"
- author names, journal metadata, DOI, affiliations, reference-list items, or publishing information
- generic words that are not useful nodes by themselves, e.g. composite, behavior, results, data, method
- standalone numbers or units unless the value is a named experimental condition, e.g. 10 K measurement temperature or +50 kOe cooling field

Use only wording explicitly present in the context. Prefer canonical, concise labels of 2-6 words. Preserve symbols that disambiguate the term, such as Hcool, Hex, Hc, Ms, CoO, or [Co/Pd0.6 nm]7.
Return only bullet lines beginning with "- ". Do not add headings, categories, explanations, markdown emphasis, or numbering.
"""


prompt_evidence_triplets = """
Given this evidence-aware scientific context:
===
{CONTEXT}
===

And this vocabulary:
===
{VOCABULARY}
===

Extract explicit triplets in the format:
- subject > relationship > object

Use only information explicitly present in the context. Do not infer missing values.
Figure captions may provide methodological or experimental context. Equations may define metrics, objectives, model components, or experimental quantities. Do not infer values from figure images or plots.
Prefer these relationship labels when they fit the context:
evaluated_on, uses_method, uses_dataset, uses_metric, reports_metric, has_value, has_unit, has_setting, has_hyperparameter, compared_with, outperforms, underperforms, improves_over, removes_component, adds_component, causes_performance_drop, causes_performance_gain, measured_under, reported_in_table.

Subjects should be vocabulary terms. Objects should be vocabulary terms, or literal values only when the literal value appears exactly in text, tables, captions, or equations. Keep relationships short and normalized. Output a list that starts with '-'.
"""


prompt_table_terms = """
Given this table-aware scientific context:

===
{text}
===

Extract only compact ontology node labels that are explicitly grounded in the table context.
Prioritize row entities, column names that are real metrics/settings, methods, materials, tasks, conditions, units, and named result types.

Do NOT output table numbers, figure numbers, table section headings, full captions, full sentences, or generic labels such as value, result, row, column, table.
Use concise labels of 2-6 words when possible. Return only bullet lines beginning with "- ".
"""


prompt_table_triplets = """
Given this table-aware scientific context:
===
{CONTEXT}
===

And this vocabulary:
===
{VOCABULARY}
===

Extract explicit table-grounded triplets in the format:
- subject > relationship > object

Use only information explicitly present in the context. Do not infer missing values.
Prefer these relationship labels when they fit the context:
reported_in_table, evaluated_on, uses_dataset, uses_metric, reports_metric, has_value, has_unit, has_setting, has_hyperparameter, compared_with, outperforms, underperforms, improves_over, removes_component, adds_component, causes_performance_drop, causes_performance_gain, measured_under.

Subjects should be vocabulary terms. Objects should be vocabulary terms, or literal values only when the literal value appears exactly in the context. Output a list that starts with '-'.
"""
