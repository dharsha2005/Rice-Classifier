"""
Insert the missing Results slide at position 11 (after Model Training).
"""
from pptx import Presentation
from pptx.oxml.ns import qn
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
import build_kec_ppt as B
from pptx.util import Inches

OUT = Path("c:/Rice classifier final project/27PR27_Rice_Classifier_Presentation.pptx")

prs = Presentation(str(OUT))
print(f"Loaded: {len(prs.slides)} slides")

# Add the Results slide at the END first (that's how python-pptx works)
B.cslide(prs, "Results & Performance Metrics", 11, B.s_results)
print("Added Results slide at end")

# Now move it to position 10 (0-indexed = slide 11, after Model Training at index 9)
# The newly added slide is at the last position
sldIdLst = prs.slides._sldIdLst
last_elem = sldIdLst[-1]   # the Results slide we just added
sldIdLst.remove(last_elem)
sldIdLst.insert(10, last_elem)  # insert at position 10 (slide 11)
print("Moved Results to position 11")

# Update slide numbers in headers
# Slides 11-16 need their slide-number text boxes updated
# (They already have the correct number embedded since we built them with
#  the right num argument, so only the moved slide has the right number)
# Actually the header text was set at build time, so they're all correct already.

print(f"Final: {len(prs.slides)} slides")
for i, s in enumerate(prs.slides):
    txts = [sh.text_frame.text[:40].replace('\n', ' ').strip()
            for sh in s.shapes if sh.has_text_frame and sh.text_frame.text.strip()]
    imgs = sum(1 for sh in s.shapes if sh.shape_type == 13)
    first = txts[0] if txts else "[empty]"
    print(f"  Slide {i+1:2d}: {first[:45]!r}  imgs={imgs}")

prs.save(str(OUT))
print("Saved OK.")
