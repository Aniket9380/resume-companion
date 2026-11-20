import hashlib
import io
import json
import os
import re
import sqlite3
from datetime import date
from pathlib import Path
from xml.sax.saxutils import escape

import streamlit as st
from docx import Document
from openai import OpenAI
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate


# ----------------------------
# App setup and configuration
# ----------------------------

st.set_page_config(
    page_title="Resume Companion",
    page_icon="🧭",
    layout="wide",
)

st.title("🧭 Resume Companion")
st.caption(
    "Tailor your resume to a job description, review every change, "
    "and download a PDF. ATS results are never guaranteed."
)

# Read the API key from Streamlit secrets when deployed.
# The sidebar input is also useful when running the app locally.
try:
    saved_api_key = st.secrets.get("OPENAI_API_KEY", "")
    saved_model = st.secrets.get("OPENAI_MODEL", "gpt-4.1-mini")
except Exception:
    saved_api_key = ""
    saved_model = "gpt-4.1-mini"

with st.sidebar:
    st.header("AI settings")

    api_key = st.text_input(
        "OpenAI API key (optional)",
        value=saved_api_key,
        type="password",
    )

    model = st.text_input(
        "Model",
        value=saved_model,
    )

    st.caption(
        "When you ask for an AI draft, your resume and job description "
        "are sent to the OpenAI API."
    )


# ----------------------------
# Resume reading and analysis
# ----------------------------

def read_resume(filename, file_bytes):
    """Read a PDF, Word document, or text resume."""
    extension = Path(filename).suffix.lower()

    if extension == ".pdf":
        pdf = PdfReader(io.BytesIO(file_bytes))
        return "\n".join(
            page.extract_text() or ""
            for page in pdf.pages
        ).strip()

    if extension == ".docx":
        document = Document(io.BytesIO(file_bytes))
        lines = [
            paragraph.text
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        ]

        # Include text that may be stored in Word tables.
        for table in document.tables:
            for row in table.rows:
                lines.append(
                    " | ".join(cell.text.strip() for cell in row.cells)
                )

        return "\n".join(lines).strip()

    if extension == ".txt":
        return file_bytes.decode(
            "utf-8",
            errors="replace",
        ).strip()

    raise ValueError("Please upload a PDF, DOCX, or TXT file.")


def get_words(text):
    """Find words for a basic keyword comparison."""
    ignored_words = {
        "with", "from", "that", "this", "will", "your", "have",
        "work", "role", "team", "ability", "years", "experience",
        "strong", "required", "preferred", "including", "using",
        "and", "the", "for", "you", "our",
    }

    words = re.findall(
        r"[A-Za-z][A-Za-z0-9+#./-]{1,}",
        text,
    )

    return {
        word.lower()
        for word in words
        if word.lower() not in ignored_words
    }


def find_possible_gaps(resume, job_description):
    """Find job description terms that don't appear in the resume."""
    resume_words = get_words(resume)
    job_words = get_words(job_description)

    return sorted(job_words - resume_words)[:35]


def check_job_posting(job_description):
    """Flag a few job-posting patterns that may need a closer look."""
    warnings = []
    text = job_description.lower()

    if len(job_description.split()) < 100:
        warnings.append(
            "This is a short posting. It may not describe the role clearly."
        )

    years = re.findall(r"\b\d+\+?\s+years?\b", text)

    if len(years) >= 3:
        warnings.append(
            "The post lists several experience requirements. "
            "Check whether they make sense together."
        )

    phrases_to_check = [
        "unlimited earning",
        "urgent hiring",
        "no experience required",
        "work from anywhere",
    ]

    if any(phrase in text for phrase in phrases_to_check):
        warnings.append(
            "The post uses generic urgency or earnings language. "
            "Verify the employer and listing."
        )

    if not warnings:
        warnings.append(
            "No obvious wording warning was found. This doesn't prove "
            "that the posting is genuine or current."
        )

    return warnings


def guess_ats(job_url):
    """Guess the application system from the job URL."""
    url = job_url.lower()

    if "myworkdayjobs.com" in url or "workday.com" in url:
        return "Workday", "Use a simple, single-column resume and check the application fields."

    if "greenhouse.io" in url:
        return "Greenhouse", "Use standard headings and review the parsed fields."

    if "lever.co" in url:
        return "Lever", "Use a simple layout and verify your details after uploading."

    if "taleo" in url or "oraclecloud.com" in url:
        return "Taleo / Oracle", "Use a plain layout and check the application fields."

    return "Unknown", "Use a simple layout and review the application preview."


def check_keyword_stuffing(text):
    """Warn about repetitive wording or possible hidden keyword text."""
    words = re.findall(r"[\w+#.-]+", text.lower())
    warnings = []

    if words:
        repeat_rate = (len(words) - len(set(words))) / len(words)

        if repeat_rate > 0.38:
            warnings.append(
                "Many words are repeated. Check that the resume sounds natural."
            )

    hidden_text_pattern = (
        r"(?:white|#fff|ffffff).*?(?:keyword|text)|"
        r"(?:keyword|text).*?(?:white|#fff|ffffff)"
    )

    if re.search(hidden_text_pattern, text, re.IGNORECASE | re.DOTALL):
        warnings.append(
            "Possible hidden or white keyword text found. Remove it."
        )

    if not warnings:
        warnings.append(
            "No obvious stuffing pattern found. Keep keywords accurate "
            "and natural."
        )

    return warnings


# ----------------------------
# AI resume draft
# ----------------------------

def make_ai_draft(resume, job_description, confirmed_experience, api_key, model):
    """Ask AI to tailor the resume without inventing candidate details."""
    if not api_key:
        return {
            "resume": resume,
            "changes": [],
            "unconfirmed_skills": find_possible_gaps(
                resume,
                job_description,
            ),
            "cover_letter": "Add an API key to draft a cover letter.",
            "linkedin_headline": "Add an API key to draft a headline.",
            "recruiter_message": "Add an API key to draft a recruiter message.",
            "interview_questions": [],
        }

    prompt = f"""
You are a careful resume editor.

Tailor the resume to the job description. Do not invent skills, employers,
dates, education, achievements, or numbers. Reorganize and rephrase only
claims supported by the source resume.

Only mention new skills if the candidate confirmed them below.
Do not exaggerate the candidate's experience.

Use standard resume headings and a simple, single-column-friendly layout.

Return only valid JSON with these keys:
- resume: complete tailored resume as plain text
- changes: list of objects with before, after, and reason
- unconfirmed_skills: possible gaps the candidate has not confirmed
- cover_letter: matching cover letter draft
- linkedin_headline: headline suggestion
- recruiter_message: short outreach draft
- interview_questions: list of likely questions

Candidate-confirmed experience:
{json.dumps(confirmed_experience)}

Current resume:
{resume[:18000]}

Job description:
{job_description[:12000]}
"""

    client = OpenAI(api_key=api_key)

    response = client.responses.create(
        model=model,
        input=prompt,
    )

    response_text = response.output_text.strip()
    response_text = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        response_text,
    )

    result = json.loads(response_text)

    # Add a default for any field the model leaves out.
    result.setdefault("resume", resume)
    result.setdefault("changes", [])
    result.setdefault("unconfirmed_skills", [])
    result.setdefault("cover_letter", "")
    result.setdefault("linkedin_headline", "")
    result.setdefault("recruiter_message", "")
    result.setdefault("interview_questions", [])

    return result


# ----------------------------
# PDF creation and readback
# ----------------------------

SECTION_NAMES = {
    "summary": "Summary",
    "experience": "Experience",
    "work experience": "Experience",
    "employment": "Experience",
    "education": "Education",
    "skills": "Skills",
    "technical skills": "Skills",
    "projects": "Projects",
    "certifications": "Certifications",
}


def create_resume_pdf(resume_text):
    """Create a simple, single-column PDF resume."""
    output = io.BytesIO()

    pdf = SimpleDocTemplate(
        output,
        pagesize=LETTER,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
        title="Tailored Resume",
    )

    styles = getSampleStyleSheet()

    name_style = ParagraphStyle(
        "ResumeName",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=17,
        leading=20,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#17324d"),
        spaceAfter=8,
    )

    heading_style = ParagraphStyle(
        "ResumeHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=13,
        textColor=colors.HexColor("#17324d"),
        spaceBefore=9,
        spaceAfter=3,
    )

    body_style = ParagraphStyle(
        "ResumeBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.2,
        leading=12,
        spaceAfter=3,
    )

    story = []
    lines = [
        line.strip()
        for line in resume_text.splitlines()
        if line.strip()
    ]

    for index, line in enumerate(lines):
        safe_line = escape(line).replace("•", "&#8226;")
        heading = line.strip().rstrip(":").lower()

        if index == 0 and heading not in SECTION_NAMES:
            story.append(Paragraph(safe_line, name_style))
        elif heading in SECTION_NAMES:
            section = SECTION_NAMES[heading].upper()
            story.append(Paragraph(escape(section), heading_style))
        else:
            story.append(Paragraph(safe_line, body_style))

    pdf.build(story)
    return output.getvalue()


def read_pdf_text(pdf_bytes):
    """Read the generated PDF back to check its extracted text."""
    pdf = PdfReader(io.BytesIO(pdf_bytes))

    return "\n".join(
        page.extract_text() or ""
        for page in pdf.pages
    ).strip()


# ----------------------------
# Simple application tracker
# ----------------------------

DATABASE_FILE = Path("applications.sqlite3")


def create_tracker_table():
    """Create the local application table if it doesn't exist."""
    with sqlite3.connect(DATABASE_FILE) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS applications (
                id INTEGER PRIMARY KEY,
                company TEXT,
                role TEXT,
                url TEXT,
                applied_on TEXT,
                outcome TEXT
            )
            """
        )


def save_application(company, role, url, outcome):
    """Save an application to the local tracker."""
    create_tracker_table()

    with sqlite3.connect(DATABASE_FILE) as connection:
        connection.execute(
            """
            INSERT INTO applications
                (company, role, url, applied_on, outcome)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                company,
                role,
                url,
                date.today().isoformat(),
                outcome,
            ),
        )


def get_applications():
    """Load saved applications from the local tracker."""
    create_tracker_table()

    with sqlite3.connect(DATABASE_FILE) as connection:
        return connection.execute(
            """
            SELECT company, role, url, applied_on, outcome
            FROM applications
            ORDER BY id DESC
            """
        ).fetchall()


# ----------------------------
# Main page
# ----------------------------

resume_file = st.file_uploader(
    "Upload your resume",
    type=["pdf", "docx", "txt"],
)

job_description = st.text_area(
    "Paste the job description",
    height=220,
)

job_url = st.text_input(
    "Job application URL (optional)",
)

resume_text = ""

if resume_file:
    try:
        resume_text = read_resume(
            resume_file.name,
            resume_file.getvalue(),
        )

        if resume_text:
            with st.expander("See extracted resume text"):
                st.text(resume_text)
        else:
            st.error(
                "I couldn't extract readable text. A scanned PDF may need OCR."
            )

    except Exception as error:
        st.error(f"I couldn't read that file: {error}")


if resume_text and job_description:
    gaps = find_possible_gaps(resume_text, job_description)

    st.subheader("1. Possible skill gaps")

    if gaps:
        st.write(", ".join(gaps))
        confirmed_skills = st.multiselect(
            "Which of these skills do you genuinely have?",
            gaps,
        )
        evidence = st.text_area(
            "Describe your experience and results for the selected skills",
        )
    else:
        confirmed_skills = []
        evidence = ""
        st.success("No possible keyword gaps were found.")

    vendor, vendor_note = guess_ats(job_url)

    left, right = st.columns(2)

    with left:
        st.markdown(f"**Likely application platform: {vendor}**")
        st.write(vendor_note)

    with right:
        st.markdown("**Job posting checks**")
        for warning in check_job_posting(job_description):
            st.write(f"• {warning}")

    if not api_key:
        st.info(
            "AI is off. You can still inspect gaps and create a PDF, "
            "but the resume won't be rewritten automatically."
        )

    if st.button("Prepare resume draft", type="primary"):
        if confirmed_skills and not evidence.strip():
            st.error(
                "Describe your experience with the selected skills, "
                "or clear your selections."
            )
        else:
            confirmed_experience = [
                f"{skill}: {evidence.strip()}"
                for skill in confirmed_skills
            ]

            try:
                with st.spinner("Preparing the draft..."):
                    draft = make_ai_draft(
                        resume_text,
                        job_description,
                        confirmed_experience,
                        api_key,
                        model,
                    )

                st.session_state["draft"] = draft
                st.session_state["resume_edit"] = draft["resume"]
                st.session_state.pop("pdf", None)
                st.session_state.pop("pdf_hash", None)

            except Exception as error:
                st.error(f"I couldn't prepare the draft: {error}")


if "draft" in st.session_state:
    draft = st.session_state["draft"]

    st.divider()
    st.subheader("2. Review the suggested changes")

    changes = draft.get("changes", [])
    decisions = []

    for number, change in enumerate(changes, start=1):
        st.markdown(
            f"**Change {number}: {change.get('reason', 'Resume edit')}**"
        )
        st.markdown("Original wording")
        st.text(change.get("before", ""))
        st.markdown("Suggested wording")
        st.text(change.get("after", ""))

        decisions.append(
            st.selectbox(
                f"Accept or reject change {number}",
                ["Accept", "Reject"],
                key=f"decision_{number}",
            )
        )

    if changes and st.button("Apply my decisions"):
        edited_resume = draft["resume"]

        for decision, change in zip(decisions, changes):
            if decision == "Reject" and change.get("after"):
                edited_resume = edited_resume.replace(
                    change["after"],
                    change.get("before", ""),
                    1,
                )

        st.session_state["resume_edit"] = edited_resume

    st.session_state.setdefault("resume_edit", draft["resume"])

    final_resume = st.text_area(
        "Final resume text",
        height=420,
        key="resume_edit",
    )

    st.caption(
        "Review names, dates, skills, and achievements before downloading."
    )

    gaps_to_review = draft.get("unconfirmed_skills", [])

    if gaps_to_review:
        st.warning(
            "Unconfirmed terms: " + ", ".join(gaps_to_review[:20])
        )

        with st.expander("Ideas for addressing these gaps honestly"):
            for skill in gaps_to_review[:12]:
                st.write(
                    f"**{skill}:** If you have related experience, describe "
                    f"the task and result. Otherwise, consider completing a "
                    f"small project using {skill} before adding it."
                )

    st.subheader("3. Supporting drafts")

    cover_tab, linkedin_tab, recruiter_tab, interview_tab = st.tabs(
        ["Cover letter", "LinkedIn", "Recruiter message", "Interview prep"]
    )

    with cover_tab:
        st.text_area(
            "Cover letter draft",
            draft.get("cover_letter", ""),
            height=200,
        )

    with linkedin_tab:
        st.text_area(
            "LinkedIn headline",
            draft.get("linkedin_headline", ""),
        )

    with recruiter_tab:
        st.text_area(
            "Recruiter message",
            draft.get("recruiter_message", ""),
            height=150,
        )

    with interview_tab:
        for question in draft.get("interview_questions", []):
            st.write(f"• {question}")

    st.subheader("4. Keyword stuffing check")

    for warning in check_keyword_stuffing(final_resume):
        st.write(f"• {warning}")

    if st.button("Create and inspect the PDF"):
        pdf_bytes = create_resume_pdf(final_resume)

        st.session_state["pdf"] = pdf_bytes
        st.session_state["pdf_text"] = read_pdf_text(pdf_bytes)
        st.session_state["pdf_hash"] = hashlib.sha256(
            final_resume.encode("utf-8")
        ).hexdigest()

    current_hash = hashlib.sha256(
        final_resume.encode("utf-8")
    ).hexdigest()

    if (
        "pdf" in st.session_state
        and st.session_state.get("pdf_hash") == current_hash
    ):
        st.download_button(
            "Download tailored_resume.pdf",
            st.session_state["pdf"],
            file_name="tailored_resume.pdf",
            mime="application/pdf",
        )

        st.subheader("5. PDF text readback")

        extracted_text = st.session_state["pdf_text"]
        st.write(f"Words extracted: {len(extracted_text.split())}")

        with st.expander("See the text extracted from the PDF"):
            st.text(extracted_text)

        st.subheader("6. Save this application")

        company = st.text_input("Company")
        role = st.text_input("Role")
        outcome = st.selectbox(
            "Application outcome",
            ["Applied", "Rejected", "Callback", "Interview", "Offer"],
        )

        if st.button("Save application"):
            save_application(company, role, job_url, outcome)
            st.success("Saved in the local application tracker.")


st.divider()
st.subheader("Application tracker")

applications = get_applications()

if applications:
    st.dataframe(
        [
            {
                "Company": row[0],
                "Role": row[1],
                "URL": row[2],
                "Date": row[3],
                "Outcome": row[4],
            }
            for row in applications
        ],
        use_container_width=True,
    )
else:
    st.caption(
        "No applications saved yet. Hosted app storage can reset, so keep "
        "your own copy of important records."
    )

st.caption(
    "ATS checks here are basic text and formatting checks. They do not "
    "reproduce a company's private ranking system."
)
