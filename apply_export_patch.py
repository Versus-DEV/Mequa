import pathlib
import shutil
import sys

ROOT = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(".")
SHELL = ROOT / "mequa" / "ui" / "shell.py"
REQS = ROOT / "requirements.txt"

IMPORT_ANCHOR = "from mequa import chat as chat_mod\n"
IMPORT_LINE = "from mequa import export_chat\n"

BLOCK_ANCHOR = (
    '                if st.button("Clear conversation", width="stretch"):\n'
    '                    ss["chat"] = []\n'
    '                    st.rerun()\n'
)
BLOCK = '''                st.markdown("**Export for Meta AI**")
                has_chat = bool(ss["chat"])
                md_text = export_chat.build_markdown(ss["chat"], ds, ss["chat_model"], ss["chat_limit"]) if has_chat else ""
                st.download_button("Download PDF", export_chat.build_pdf(md_text) if has_chat else b"",
                                   file_name="mequa_chat.pdf", mime="application/pdf",
                                   disabled=not has_chat, width="stretch", key="export_pdf")
                st.download_button("Download Markdown", md_text, file_name="mequa_chat.md",
                                   mime="text/markdown", disabled=not has_chat, width="stretch", key="export_md")
                if has_chat:
                    with st.expander("Paste-ready text"):
                        st.code(export_chat.build_prompt(ss["chat"], ds, ss["chat_model"], ss["chat_limit"]), language=None)
'''


def patch_shell() -> str:
    if not SHELL.is_file():
        return f"MISSING: {SHELL}"
    text = SHELL.read_text(encoding="utf-8")
    if "export_chat" in text:
        return "shell.py already patched, left unchanged"
    if text.count(IMPORT_ANCHOR) != 1 or text.count(BLOCK_ANCHOR) != 1:
        return "shell.py does not match the expected layout; nothing changed. Apply the two edits by hand (see EXPORT_SETUP.md)"
    shutil.copyfile(SHELL, SHELL.with_suffix(".py.bak"))
    text = text.replace(IMPORT_ANCHOR, IMPORT_ANCHOR + IMPORT_LINE)
    text = text.replace(BLOCK_ANCHOR, BLOCK_ANCHOR + BLOCK)
    SHELL.write_text(text, encoding="utf-8")
    return "shell.py patched (backup: shell.py.bak)"


def patch_requirements() -> str:
    if not REQS.is_file():
        return f"MISSING: {REQS}"
    lines = REQS.read_text(encoding="utf-8").splitlines()
    if any(l.strip().lower().startswith("fpdf2") for l in lines):
        return "requirements.txt already lists fpdf2"
    REQS.write_text("\n".join(lines).rstrip("\n") + "\nfpdf2\n", encoding="utf-8")
    return "fpdf2 added to requirements.txt"


if __name__ == "__main__":
    print(patch_shell())
    print(patch_requirements())
