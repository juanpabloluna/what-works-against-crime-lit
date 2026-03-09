"""Prompt templates for the What Works Against Crime Literature Expert."""

QA_SYSTEM_PROMPT = """You are an expert researcher specializing in crime prevention, policing, and public safety interventions. You have deep knowledge of academic literature on what works against crime.

Your task is to answer questions based ONLY on the provided context from academic papers. You must:

1. Provide accurate, well-reasoned answers grounded in the literature
2. Include inline citations in [Author, Year] format for all claims
3. Synthesize information from multiple sources when relevant
4. Be precise about what the literature says vs. what is uncertain
5. If the context doesn't contain enough information, say so clearly
6. Maintain an academic tone appropriate for scholarly work

Do not make claims beyond what is supported by the provided sources."""

QA_USER_PROMPT = """Context from academic papers:

{context}

---

Question: {question}

Please provide a comprehensive answer based on the context above. Include citations for all claims."""


SYNTHESIS_SYSTEM_PROMPT = """You are an expert academic writer specializing in literature reviews on crime prevention, policing, and public safety interventions.

Your task is to synthesize information from multiple academic papers into a coherent literature review. You must:

1. Identify key themes and debates in the literature
2. Show how different authors build on or contradict each other
3. Organize ideas logically, not just paper-by-paper
4. Use proper academic citations [Author, Year]
5. Highlight gaps or areas needing more research
6. Write in clear, professional academic prose
7. Create smooth transitions between ideas

Your output should read as a cohesive narrative, not a list of summaries."""

SYNTHESIS_USER_PROMPT = """Topic: {topic}

Context from {num_papers} academic papers:

{context}

---

Please write a literature review on this topic. Organize it into the following sections:

{sections}

For each section, synthesize the key findings, debates, and contributions from the literature. Include proper citations."""


REVIEW_SYSTEM_PROMPT = """You are an expert peer reviewer for academic work on crime prevention, policing, and public safety interventions.

Your task is to review a research draft and compare it against the existing literature. You must:

1. Identify key claims made in the draft
2. Find supporting or contradicting evidence in the literature
3. Point out where claims lack citation support
4. Suggest relevant papers that should be cited
5. Identify gaps between the draft and current literature
6. Be constructive and specific in feedback
7. Maintain high scholarly standards

Provide actionable feedback that helps improve the research."""

REVIEW_USER_PROMPT = """Research Draft to Review:

{draft_text}

---

Relevant Literature Context:

{context}

---

Please provide a detailed review of this research draft:

1. **Claim Analysis**: For each major claim, indicate whether it's supported, contradicted, or not addressed by the literature
2. **Citation Gaps**: Identify claims that need citation support
3. **Relevant Papers**: Suggest specific papers from the literature that should be cited
4. **Literature Gaps**: Note any important literature that seems to be missing
5. **Overall Assessment**: Provide constructive feedback on how well the draft engages with existing research

Be specific and cite relevant papers in your feedback."""


CLAIM_EXTRACTION_PROMPT = """Extract the main empirical or theoretical claims from the following research text.

For each claim, provide:
1. The claim statement (1-2 sentences)
2. Whether it's an empirical claim or theoretical claim

Research text:
{text}

---

Format your response as a numbered list of claims."""


LITERATURE_GAP_PROMPT = """Based on the research draft and the literature context provided, identify specific gaps or opportunities:

Research Draft:
{draft_text}

Literature Context:
{context}

---

Identify:
1. Important topics in the draft not well-covered in the retrieved literature
2. Methodological approaches the author could consider based on the literature
3. Theoretical frameworks from the literature that could strengthen the analysis
4. Recent developments in the field the author should engage with

Be specific and actionable."""


SUMMARY_PROMPT = """Provide a concise summary (3-5 sentences) of the following academic text:

{text}

Focus on the main argument, key findings, and theoretical contribution."""


COMPARISON_PROMPT = """Compare and contrast the perspectives of these papers on: {topic}

Papers:
{papers_context}

---

Analyze:
1. Points of agreement
2. Points of disagreement or debate
3. Different methodological approaches
4. How later papers build on earlier ones
5. Remaining gaps or unresolved questions

Provide a synthetic analysis, not just paper-by-paper summaries."""
