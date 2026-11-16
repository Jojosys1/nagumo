from pypdf import PdfReader
from pathlib import Path
for pdf in [Path('cahn_hilliard_modeling (1).pdf'), Path('Problème_instationnaire_Nagumo_s_model.pdf')]:
    reader = PdfReader(str(pdf))
    out = pdf.with_suffix('.txt')
    chunks=[]
    for i,p in enumerate(reader.pages, start=1):
        txt = p.extract_text() or ''
        chunks.append(f"\n\n===== PAGE {i} =====\n\n{txt}")
    out.write_text(''.join(chunks), encoding='utf-8')
    print(f"wrote {out} ({len(reader.pages)} pages)")
