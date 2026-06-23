import re
import sys

tex = open(sys.argv[1], encoding="utf-8", errors="ignore").read()
inc = re.findall(r"\\includegraphics[^{]*\{([^}]+)\}", tex)
print("TOTAL includegraphics:", len(inc))
print("unique:", len(set(inc)))
print()
# figure blocks: \begin{figure}...\end{figure}
for blk in re.findall(r"\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}", tex, re.S):
    files = re.findall(r"\\includegraphics[^{]*\{([^}]+)\}", blk)
    cap = re.search(r"\\caption\{(.*?)\}", blk, re.S)
    lab = re.search(r"\\label\{(.*?)\}", blk)
    cap_txt = re.sub(r"\s+", " ", cap.group(1))[:160] if cap else "(no caption)"
    names = ", ".join(f.split("/")[-1] for f in files)
    print(f"[{names}]  {lab.group(1) if lab else ''}")
    print(f"   {cap_txt}")
