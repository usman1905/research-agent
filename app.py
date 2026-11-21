"""WebResearchAI web app. Run with: streamlit run app.py"""

import re

import streamlit as st

from webresearch import ResearchAgent
from webresearch.llm import LLM, LLMError
from webresearch.web import domain

LOGO = "https://usman1905.github.io/logos/webresearchai-wordmark.webp"
ICON = "https://usman1905.github.io/logos/webresearchai.webp"
EXAMPLES = [
    "What are the main differences between 304 and 316L stainless steel for jewellery?",
    "How does retrieval-augmented generation reduce hallucinations?",
    "What is the current population of Pakistan?",
]

st.set_page_config(page_title="WebResearchAI", page_icon=":material/travel_explore:", layout="centered")
st.logo(ICON, link="https://github.com/usman1905/research-agent")


def secrets() -> dict:
    try:
        return dict(st.secrets)
    except Exception:  # no secrets file locally
        return {}


def linked(answer: str, urls: dict[int, str]) -> str:
    """Turn [1] and [1, 2] into links to the sources."""

    def link(m: re.Match) -> str:
        ids = [int(x) for x in m.group(1).split(",")]
        return " ".join(f"[[{i}]]({urls[i]})" for i in ids if i in urls)

    return re.sub(r"\[(\d+(?:\s*,\s*\d+)*)\]", link, answer)


with st.sidebar:
    st.subheader("Settings")
    max_pages = st.slider("Pages to read", 3, 10, 6)
    st.caption(
        "WebResearchAI plans searches, reads the pages it finds, picks the most relevant passages and "
        "answers with citations. Every citation is then checked against the source text."
    )
    st.caption("[Source code](https://github.com/usman1905/research-agent) · Built by [Usman](https://usman1905.github.io)")

st.image(LOGO, width=300)
st.write("Ask a question. I search the web, read the sources and answer with citations you can check.")

try:
    llm = LLM.from_env(secrets())
except LLMError as e:
    st.warning(f"{e} Add it under Settings → Secrets on Streamlit Cloud, or in `.streamlit/secrets.toml` locally.")
    st.stop()

if "question" not in st.session_state:
    st.session_state.question = ""

cols = st.columns(len(EXAMPLES))
for col, example in zip(cols, EXAMPLES, strict=True):
    if col.button(example, width="stretch"):
        st.session_state.question = example

with st.form("ask"):
    question = st.text_area("Your question", key="question", height=90, placeholder="What do you want to research?")
    submitted = st.form_submit_button("Research", type="primary", width="stretch")

if submitted and question.strip():
    agent = ResearchAgent(llm, max_pages=max_pages)
    with st.status("Researching…", expanded=True) as status:
        try:
            result = agent.run(question, step=st.write)
        except (LLMError, ValueError) as e:
            status.update(label="Research failed", state="error")
            st.error(str(e))
            st.stop()
        status.update(label="Done", state="complete", expanded=False)

    urls = {s.id: s.url for s in result.sources}
    st.subheader("Answer")
    st.markdown(linked(result.answer, urls))

    cited = [c for c in result.checks if c.status != "uncited"]
    if cited:
        supported = sum(c.status == "supported" for c in cited)
        st.caption(f"Citation check: {supported} of {len(cited)} cited sentences are backed by their sources.")
        weak = [c for c in cited if c.status != "supported"]
        if weak:
            with st.expander(f"{len(weak)} sentence{'s' if len(weak) != 1 else ''} to double-check"):
                for c in weak:
                    st.markdown(f"- {c.sentence}  \n  *{c.support:.0%} of its key words appear in the cited source.*")

    if result.sources:
        st.subheader("Sources")
        for s in result.sources:
            st.markdown(f"**[{s.id}]** [{s.title}]({s.url})  \n{domain(s.url)}")
    with st.expander("Searches I ran"):
        for q in result.queries:
            st.markdown(f"- {q}")
