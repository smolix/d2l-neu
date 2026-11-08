"""Build Arabic, Turkish and German terminology from inspected source alignments.

This helper is deliberately conservative: it marks a candidate verified only when
the exact English concept is a paired glossary head or paired D2L heading/emphasis.
The output still needs the manual sense audit documented in notes/ar-tr-de.md.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from html import unescape
import json
from pathlib import Path
import re
import unicodedata


BASE = Path(__file__).resolve().parent
CORPORA = Path("/tmp/d2l-glossary-corpora")
GOOGLE = CORPORA / "google-topics"
TOPICS = "fundamentals agent clustering df generative googlecloud metrics responsible-ai tensorflow".split()
LOCALES = ("ar", "tr", "de")
COMMITS = {"ar": "71cbd6268af46f447805d8f508686745b71aedd9", "tr": "fff29a8aa5dd6dcab30e45caff912a5b290286f6", "de": "c4b458ff1059f4a0aca8e58e12d2970d11a6298f"}
CHEAT_COMMIT = "99f45b6d65b8e60f637d8ac0db15f76327197c23"
REPOSITORIES = {"ar": "mohamed-ali/d2l-ar", "tr": "d2l-ai/d2l-tr", "de": "d2l-ai/d2l-de"}
TAIMA_PDF_IDS = {1: "1SrmT_r8dNK42IqyS0gwXtbLCZbk5G8eu", 2: "1UqEu0amRfAvJD0L1HosIn3UJi0FkNemU", 3: "1-HlY2IqKIIEKfFrxebBu51z9oAPftLfA"}

# Senses where a homographic Google head often uses a different definition or
# an awkward machine-rendered word. They remain candidates for targeted review.
GOOGLE_DENY = {
    "ar": {"label", "precision", "recall", "kernel", "score-function", "statistical-bias", "return", "state", "policy", "overfitting", "underfitting", "regularization", "context-window", "ground-truth", "sparsity", "training-loss", "validation-loss"},
    "tr": {"epoch", "precision", "recall", "kernel", "score-function", "statistical-bias", "return", "state", "policy", "logistic-regression", "action", "implicit-bias", "one-hot-encoding"},
    "de": {"prediction", "recall", "kernel", "score-function", "statistical-bias", "return", "state", "policy", "confusion-matrix", "sparsity", "shrinkage"},
}
D2L_DENY = {
    "tr": {"linear-transformation", "sampling", "max-pooling", "language-model", "long-short-term-memory", "intersection-over-union", "byte-pair-encoding", "feature-map", "sequence-transduction", "affine-transformation", "underflow"},
    "de": {"regression", "dropout", "surprisal", "reinforcement-learning", "chain-rule", "weight-decay", "forward-propagation", "stochastic-gradient-descent", "computational-graph", "vanishing-gradient", "exploding-gradient", "covariate-shift", "label-shift", "joint-probability"},
}
PREFERRED_NORMALIZE = {
    "tr": {"expectation": "beklenti", "derivative": "türev", "partial-derivative": "kısmi türev", "forward-pass": "ileri geçiş", "gram-matrix": "Gram matrisi", "overflow": "taşma", "projection": "izdüşüm", "likelihood": "olabilirlik", "probability": "olasılık", "training-loss": "eğitim kaybı", "forward-propagation": "ileri yayma", "epoch": "dönem"},
    "de": {"stochastic-gradient-descent": "stochastischer Gradientenabstieg"},
}
TERMINOLOGY_DENY = {"aliasing", "imputation", "one-hot-encoding", "permutation", "tensor", "underflow"}
TERMINOLOGY_PREFERRED = {
    "regression": "regresyon", "bias": "ek girdi", "statistical-bias": "yanlılık",
    "loss-function": "kayıp fonksiyonu", "machine-learning": "makine öğrenmesi",
    "overfitting": "aşırı öğrenme", "token": "belirteç", "softmax": "softmaks",
    "variance": "değişinti", "feature": "öznitelik", "regularization": "düzenlileştirme",
    "dropout": "hattan düşme", "epoch": "dönem", "likelihood": "olabilirlik",
    "validation-set": "geçerleme kümesi", "computer-vision": "bilgisayarla görme",
}
KIT_DENY = {"feature", "ground-truth", "reinforcement-learning", "target", "perceptor", "instance"}
CHEAT_DENY = {"ar": {"bias", "gradient"}, "tr": {"bias", "statistical-bias", "objective-function"}, "de": {"dropout", "long-short-term-memory", "batch-normalization"}}

# Reviewed on the source PDF text layer, with representative pages rendered to
# check that the extracted Arabic reads in the intended direction. Each item is
# preferred, volume, PDF page, exact short expression in the extracted page.
# The PDF layer sometimes duplicates an initial alif; the preferred form uses
# the visually correct spelling while source_term preserves the extracted one.
AR_PDF_APPROVED = {
    "overfitting": ("الضبط الزائد", 1, 164, "الضبط الزائد"),
    "underfitting": ("الضبط الناقص", 1, 164, "الضبط الناقص"),
    "regularization": ("التنظيم", 1, 175, "التنظيم"),
    "regression": ("الانحدار", 1, 124, "االنحدار"),
    "classification": ("التصنيف", 1, 39, "التصنيف"),
    "label": ("التسمية", 1, 32, "التسمية"),
    "dataset": ("مجموعة البيانات", 1, 124, "مجموعة البيانات"),
    "test-set": ("مجموعة الاختبار", 1, 35, "مجموعة االختبار"),
    "empirical-risk": ("الخطر التجريبي", 1, 219, "الخطر التجريبي"),
    "minibatch": ("دفعة صغيرة", 1, 129, "دفعة صغيرة"),
    "stochastic-gradient-descent": ("التدرج الاشتقاقي العشوائي", 1, 57, "التدرج االشتقاقي العشوائي"),
    "optimization": ("التحسين", 1, 98, "التحسين"),
    "gradient": ("التدرج", 1, 97, "تدرج الدالة"),
    "derivative": ("المشتق", 1, 92, "المشتق"),
    "partial-derivative": ("المشتق الجزئي", 1, 97, "المشتق الجزئي"),
    "chain-rule": ("قاعدة السلسلة", 1, 98, "قاعدة السلسلة"),
    "automatic-differentiation": ("التفاضل التلقائي", 1, 99, "التفاضل التلقائي"),
    "computational-graph": ("الرسم البياني الحسابي", 1, 103, "الرسم البياني الحسابي"),
    "linear-algebra": ("الجبر الخطي", 1, 70, "الجبر الخطي"),
    "linear-transformation": ("التحويل الخطي", 1, 235, "التحويل الخطي"),
    "identity-matrix": ("مصفوفة الوحدة", 3, 36, "مصفوفة الوحدة"),
    "random-variable": ("المتغير العشوائي", 1, 109, "المتغير العشوائي"),
    "independence": ("الاستقلال", 1, 113, "االستقالل"),
    "expectation": ("التوقع", 1, 116, "التوقع"),
    "variance": ("التباين", 1, 117, "التباين"),
    "standard-deviation": ("الانحراف المعياري", 1, 117, "االنحراف المعياري"),
    "covariance": ("التغاير", 1, 117, "التغاير"),
    "bernoulli-distribution": ("توزيع برنولي", 1, 52, "توزيع برنولي"),
    "poisson-distribution": ("توزيع بواسون", 1, 138, "توزيع بواسون"),
    "maximum-likelihood-estimation": ("تقدير الاحتمال الأقصى", 3, 318, "تقدير االحتمال األقصى"),
    "sampling": ("أخذ العينات", 3, 62, "أخذ العينات"),
    "cross-entropy-loss": ("خطأ الإنتروبيا المتقاطعة", 1, 184, "خطأ االنتروبيا المتقاطعة"),
    "confusion-matrix": ("مصفوفة الارتباك", 1, 221, "مصفوفة االرتباك"),
    "nonlinearity": ("اللاخطية", 1, 230, "الالخطية"),
    "weight-decay": ("اضمحلال الوزن", 1, 164, "اضمحالل الوزن"),
    "dropout": ("التسرب", 1, 259, "التسرب"),
    "model-selection": ("اختيار النموذج", 1, 166, "اختيار النموذج"),
    "initialization": ("التهيئة", 2, 34, "التهيئة"),
    "vanishing-gradient": ("تلاشي التدرج", 1, 249, "تالشي التدرج"),
    "exploding-gradient": ("انفجار التدرجات", 2, 210, "انفجار التدرجات"),
    "gradient-clipping": ("قص التدرج", 2, 211, "قص التدرج"),
    "batch-normalization": ("التسوية بالدفعات", 2, 128, "التسوية بالدفعات"),
    "layer-normalization": ("تسوية الطبقة", 2, 133, "تسوية الطبقة"),
    "padding": ("الحشو", 2, 70, "الحشو"),
    "pooling": ("التجميع", 2, 82, "التجميع"),
    "average-pooling": ("تجميع المتوسط", 2, 82, "تجميع المتوسط"),
    "residual-connection": ("اتصال متبقٍ", 2, 144, "اتصال متبقي"),
    "image-augmentation": ("زيادة الصور", 3, 185, "زيادة الصورة"),
    "transfer-learning": ("نقل التعلم", 3, 202, "نقل التعلم"),
    "pretraining": ("التدريب المسبق", 2, 342, "التدريب المسبق"),
    "object-detection": ("اكتشاف الكائنات", 3, 185, "اكتشاف الكائنات"),
    "bounding-box": ("مربع محيط", 3, 204, "مربعًا محيطًا"),
    "semantic-segmentation": ("التجزئة الدلالية", 3, 185, "التجزئة الداللية"),
    "image-classification": ("تصنيف الصور", 1, 179, "تصنيف الصور"),
    "computer-vision": ("الرؤية الحاسوبية", 1, 29, "الرؤية الحاسوبية"),
    "hidden-state": ("حالة مخفية", 2, 200, "حالة مخفية"),
    "language-model": ("نموذج اللغة", 2, 194, "نموذج اللغة"),
    "embedding": ("التضمين", 2, 263, "التضمين"),
    "word-embedding": ("تضمين الكلمات", 3, 315, "تضمين الكلمات"),
    "encoder": ("المشفر", 2, 258, "المشفر"),
    "decoder": ("مفكك الشفرة", 2, 258, "مفكك الشفرة"),
    "teacher-forcing": ("إجبار المعلم", 2, 262, "إجبار المعلم"),
    "beam-search": ("البحث الشعاعي", 2, 275, "البحث الشعاعي"),
    "perplexity": ("الارتباك", 2, 196, "االرتباك"),
    "attention": ("الانتباه", 2, 87, "االنتباه"),
    "multi-head-attention": ("الانتباه متعدد الرؤوس", 2, 279, "االنتباه متعدد الرؤوس"),
    "query": ("الاستعلام", 2, 87, "االستعالم"),
    "positional-encoding": ("ترميز موضعي", 2, 316, "ترميز موضعي"),
    "transformer": ("المحوّل", 2, 172, "المحوالت"),
    "reinforcement-learning": ("التعلم المعزز", 1, 49, "التعلم المعزز"),
    "convexity": ("التحدب", 3, 28, "التحدب"),
    "local-minimum": ("قيمة صغرى محلية", 3, 25, "قيمة صغرى محلية"),
    "hessian": ("المصفوفة الهيسية", 3, 28, "المصفوفة الهيسية"),
    "recommender-system": ("نظام التوصية", 2, 173, "نظام التوصية"),
    "negative-sampling": ("أخذ العينات السلبية", 3, 322, "أخذ العينات السلبية"),
    "sentiment-analysis": ("تحليل المشاعر", 3, 384, "تحليل المشاعر"),
    "principal-component-analysis": ("تحليل المكون الأساسي", 1, 47, "تحليل المكون األساسي"),
    "empirical-risk-minimization": ("تقليل الخطر التجريبي", 1, 219, "تقليل الخطر التجريبي"),
    "central-limit-theorem": ("نظرية الحد المركزي", 1, 206, "نظرية الحد المركزي"),
    "law-of-large-numbers": ("قانون الأعداد الكبيرة", 1, 107, "قانون األعداد الكبيرة"),
    "distribution-shift": ("تحول التوزيع", 1, 226, "تحول التوزيع"),
    "data-parallelism": ("توازي البيانات", 3, 159, "توازي البيانات"),
    "memory-bandwidth": ("النطاق الترددي للذاكرة", 2, 128, "النطاق الترددي للذاكرة"),
    "quantization": ("التكميم", 2, 37, "التكميم"),
    "probability-mass-function": ("دالة الكتلة الاحتمالية", 2, 174, "دالة الكتلة االحتمالية"),
    "numerical-stability": ("الاستقرار العددي", 1, 203, "االستقرار العددي"),
    "floating-point": ("الفاصلة العائمة", 1, 187, "الفاصلة العائمة"),
    "line-search": ("البحث الخطي", 3, 53, "البحث الخطي"),
    "warmup": ("الإحماء", 3, 108, "اإلحماء"),
    "forward-propagation": ("الانتشار الأمامي", 1, 243, "االنتشار األمامي"),
    "computational-efficiency": ("الكفاءة الحسابية", 3, 65, "الكفاءة الحسابية"),
    "inductive-bias": ("التحيز الاستقرائي", 2, 56, "التحيز االستقرائي"),
    "representation-learning": ("التعلم التمثيلي", 3, 384, "التعلم التمثيلي"),
    "supervised-learning": ("التعلم الخاضع للإشراف", 1, 32, "الخاضع لإلشراف"),
    "autoregressive-model": ("نموذج الانحدار الذاتي", 2, 216, "نموذج االنحدار الذاتي"),
    "causal-attention": ("انتباه سببي", 2, 346, "انتباه سببي"),
    "residual-block": ("الكتلة المتبقية", 2, 143, "الكتلة المتبقية"),
    "attention-pooling": ("تجميع الانتباه", 2, 282, "تجميع االنتباه"),
    "additive-attention": ("الانتباه الإضافي", 2, 301, "االنتباه اإلضافي"),
    "subword": ("الكلمات الفرعية", 3, 350, "الكلمات الفرعية"),
    "word-frequency": ("تكرار الكلمة", 2, 189, "تكرار الكلمة"),
    "cosine-similarity": ("تشابه جيب التمام", 3, 316, "تشابه جيب التمام"),
    "question-answering": ("الإجابة على الأسئلة", 3, 357, "اإلجابة على األسئلة"),
    "convolutional-layer": ("الطبقة التلافيفية", 2, 56, "الطبقة التالفيفية"),
    "feature-map": ("خريطة المعالم", 2, 67, "خريطة المعالم"),
    "pooling-layer": ("طبقة التجميع", 3, 253, "طبقة تجميع"),
    "transposed-convolution": ("الالتفاف المنقول", 3, 271, "االلتفاف المنقول"),
    "image-segmentation": ("تقطيع الصورة", 3, 258, "تقطيع الصورة"),
    "region-proposal": ("اقتراح المنطقة", 3, 255, "اقتراح المنطقة"),
    "data-augmentation": ("زيادة البيانات", 3, 235, "زيادة البيانات"),
    "deep-reinforcement-learning": ("التعلم المعزز العميق", 1, 49, "التعلم المعزز العميق"),
    "gaussian-noise": ("الضوضاء الغاوسية", 1, 39, "الضوضاء الغاوسية"),
    "dynamic-programming": ("البرمجة الديناميكية", 2, 177, "البرمجة الديناميكية"),
    "ridge-regression": ("انحدار ريدج", 1, 170, "انحدار ريدج"),
    "gram-matrix": ("مصفوفة جرام", 3, 285, "مصفوفة جرام"),
    "kernel": ("نواة الالتفاف", 2, 60, "نواة االلتفاف"),
    "covariance-matrix": ("مصفوفة التغاير", 1, 117, "مصفوفة التغاير"),
    "broadcasting": ("البث", 1, 74, "البث"),
    "finite-difference": ("الفروقات المحدودة", 2, 69, "الفروقات المحدودة"),
    "mean-value-theorem": ("نظرية القيمة المتوسطة", 3, 36, "نظرية القيمة المتوسطة"),
    "taylor-expansion": ("توسعة تايلور", 2, 154, "توسعة تايلور"),
    "data-loader": ("محمّل البيانات", 2, 198, "مُحمل البيانات"),
    "generalization-gap": ("فجوة التعميم", 1, 209, "فجوة التعميم"),
    "forget-gate": ("بوابة النسيان", 2, 228, "بوابة النسيان"),
    "input-gate": ("بوابة الإدخال", 2, 228, "بوابة اإلدخال"),
    "output-gate": ("بوابة الإخراج", 2, 229, "بوابة اإلخراج"),
    "reset-gate": ("بوابة إعادة الضبط", 2, 236, "بوابة إعادة الضبط"),
    "sequence-modeling": ("نمذجة التسلسل", 2, 174, "نمذجة التسلسل"),
    "memory-footprint": ("بصمة الذاكرة", 2, 80, "بصمة الذاكرة"),
    "synchronization": ("المزامنة", 3, 158, "المزامنة"),
    "quadratic-form": ("الشكل التربيعي", 1, 127, "الشكل التربيعي"),
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def norm(text: str) -> str:
    text = unescape(re.sub(r"<[^>]+>", "", text))
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", text).strip()


def key(text: str) -> str:
    return norm(text).casefold()


def google_heads(path: Path) -> dict[str, tuple[str, str]]:
    s = path.read_text(errors="replace")
    matches = list(re.finditer(r'<h2\b[^>]*\bid="([^"]+)"[^>]*>(.*?)</h2>', s, re.S | re.I))
    out = {}
    for i, m in enumerate(matches):
        anchor, term = m.group(1), norm(m.group(2))
        end = matches[i + 1].start() if i + 1 < len(matches) else min(len(s), m.end() + 4000)
        body = s[m.end():end]
        # The first few prose sentences establish the sense of this head.
        body = re.sub(r"<div class=\"glossary-icon-container\".*?</div>\s*</div>", "", body, flags=re.S)
        context = norm(body[:2000])[:230]
        out[anchor] = (term, context)
    return out


def google_index(locale: str) -> dict[str, list[dict]]:
    out = defaultdict(list)
    for topic in TOPICS:
        english = google_heads(GOOGLE / f"{topic}-en.html")
        translated = google_heads(GOOGLE / f"{topic}-{locale}.html")
        for anchor, (en, en_context) in english.items():
            if anchor not in translated:
                continue
            target, target_context = translated[anchor]
            if not target or key(en) == key(target):
                continue
            out[key(en)].append({"term": target, "topic": topic, "anchor": anchor, "en_context": en_context, "target_context": target_context})
    return out


def markdown_heads(path: Path) -> list[tuple[int, str]]:
    result = []
    in_fence = False
    for i, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        if line.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
            continue
        if not in_fence and (m := re.match(r"^(#{1,6})\s+(.+)$", line)):
            result.append((i, norm(m.group(2))))
    return result


def emphasis(path: Path) -> list[tuple[int, str]]:
    s = re.sub(r"```.*?```", lambda m: "\n" * m.group().count("\n"), path.read_text(errors="replace"), flags=re.S)
    return [(s[:m.start()].count("\n") + 1, norm(m.group(1))) for m in re.finditer(r"(?<!\*)\*([^*\n]{2,80})\*(?!\*)", s)]


def d2l_index(locale: str) -> dict[str, list[dict]]:
    root = CORPORA / f"d2l-{locale}"
    out = defaultdict(list)
    for target_file in root.glob("chapter_*/*.md"):
        if target_file.name.endswith("_origin.md"):
            continue
        original = target_file.with_name(target_file.stem + "_origin.md")
        if not original.exists():
            continue
        for kind, extractor in (("heading", markdown_heads), ("emphasis", emphasis)):
            en_items, target_items = extractor(original), extractor(target_file)
            if len(en_items) != len(target_items):
                continue
            for (_, en), (line, target) in zip(en_items, target_items):
                if not en or not target or key(en) == key(target):
                    continue
                if len(target.split()) > 10 or len(target) > 100:
                    continue
                out[key(en)].append({"term": target, "path": target_file.relative_to(root).as_posix(), "line": line, "kind": kind})
    return out


def terminology_index() -> dict[str, list[dict]]:
    path = CORPORA / "d2l-tr" / "TERMINOLOGY.md"
    out = defaultdict(list)
    for line, text in enumerate(path.read_text().splitlines(), 1):
        if ";" not in text:
            continue
        en, target = [norm(x) for x in text.split(";", 1)]
        if en and target:
            out[key(en)].append({"term": target, "path": "TERMINOLOGY.md", "line": line, "kind": "curated glossary"})
    return out


def kit_index() -> dict[str, list[dict]]:
    pages = (CORPORA / "de-kit-glossary.txt").read_text().split("\f")
    out = defaultdict(list)
    for page in (53, 54):
        for line in pages[page - 1].splitlines():
            match = re.match(r"^\s*(\S.*?)\s{3,}(\S.*?)\s*$", line)
            if not match:
                continue
            en, target = (norm(x) for x in match.groups())
            if en and target and en.casefold() != target.casefold():
                out[key(en)].append({"term": target, "page": page, "kind": "bilingual academic glossary"})
    return out


def cheat_index(locale: str) -> dict[str, list[dict]]:
    root = CORPORA / "cheatsheet-translation" / locale
    out = defaultdict(list)
    for path in root.glob("*.md"):
        s = path.read_text(errors="replace")
        items = list(re.finditer(r"^\*\*(\d+)\.\s+(.+?)\*\*\s*$", s, re.M))
        for i, item in enumerate(items):
            block = s[item.end():items[i + 1].start() if i + 1 < len(items) else len(s)]
            english = norm(re.split(r"\s+[―–]\s+|\s+-\s+", item.group(2), 1)[0]).strip(":[]")
            translated = None
            translated_line = None
            for line_no_offset, line in enumerate(block.splitlines(), 1):
                if "&#10230;" in line:
                    translated = line.split("&#10230;", 1)[1].strip()
                    translated_line = line_no_offset
                    break
                if locale == "ar" and re.search(r"[\u0600-\u06FF]", line):
                    translated = line.strip()
                    translated_line = line_no_offset
                    break
            if not translated:
                continue
            translated = norm(translated)
            source_term = re.split(r"\s+[―–]\s+|\s+-\s+", translated, 1)[0].strip(":[]")
            preferred = re.sub(r"\s*\([A-Za-z][^)]*\)$", "", source_term).strip()
            if not preferred or len(preferred) > 70 or len(preferred.split()) > 8 or key(preferred) == key(english):
                continue
            line_no = s[:item.end()].count("\n") + translated_line
            out[key(english)].append({"term": preferred, "source_term": source_term,
                                      "path": f"{locale}/{path.name}", "line": line_no, "kind": "paired teaching sheet"})
    return out


def editorial_index() -> dict[str, dict[str, str]]:
    rows = (BASE / "candidates" / "editorial-ar-tr-de.tsv").read_text().splitlines()
    result = {}
    for line in rows[1:]:
        if not line.strip():
            continue
        cid, ar, tr, de = line.split("\t")
        result[cid] = {"ar": ar, "tr": tr, "de": de}
    return result


def prose_occurrences(locale: str) -> list[tuple[str, str, int, str]]:
    """Text-only lines for candidate lookup, excluding English originals and code."""
    items = []
    root = CORPORA / f"d2l-{locale}"
    for path in root.glob("chapter_*/*.md"):
        if path.name.endswith("_origin.md"):
            continue
        in_fence = False
        for line_no, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            if line.lstrip().startswith(("```", "~~~")):
                in_fence = not in_fence
                continue
            if not in_fence and line.strip() and not line.lstrip().startswith(("# `", "!", "[", ":", "<!--", "|")):
                items.append(("d2l", path.relative_to(root).as_posix(), line_no, line))
    cheat_root = CORPORA / "cheatsheet-translation" / locale
    for path in cheat_root.glob("*.md"):
        for line_no, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            if line.strip() and ("&#10230;" in line or (locale == "ar" and re.search(r"[\u0600-\u06FF]", line))):
                items.append(("cheat", f"{locale}/{path.name}", line_no, line))
    return items


def find_prose_occurrence(term: str, items: list[tuple[str, str, int, str]]) -> tuple[str, str, int, str] | None:
    pattern = re.compile(r"(?<!\w)" + re.escape(term) + r"(?!\w)", re.I)
    for kind, path, line_no, line in items:
        if pattern.search(line):
            return kind, path, line_no, line
    return None


def rows_for(locale: str, concepts: list[dict]) -> tuple[list[dict], list[dict]]:
    gd = google_index(locale)
    dd = d2l_index(locale) if locale in {"tr", "de"} else {}
    tt = terminology_index() if locale == "tr" else {}
    kk = kit_index() if locale == "de" else {}
    cc = cheat_index(locale)
    editorial = editorial_index()
    prose = prose_occurrences(locale)
    sources = [{
        "id": f"d2l-{locale}-source", "title": f"Dive into Deep Learning, {locale} source",
        "locale": locale, "kind": "earlier D2L translation repository" if locale != "ar" else "independent Arabic translation repository",
        "url": f"https://github.com/{REPOSITORIES[locale]}", "accessed": str(date.today()),
        "publisher": "D2L.ai community" if locale != "ar" else "mohamed-ali",
        "repository": REPOSITORIES[locale], "commit": COMMITS[locale],
        "notes": "Automated English originals ending _origin.md excluded. German is partial/old and contains machine translation errors." if locale == "de" else "Check sense and translation quality; older D2L edition.",
    }]
    if locale == "tr":
        sources.append({"id": "d2l-tr-terminology", "title": "D2L Turkish editorial terminology list", "locale": "tr",
                        "kind": "bilingual project terminology list", "url": f"https://github.com/d2l-ai/d2l-tr/blob/{COMMITS['tr']}/TERMINOLOGY.md",
                        "accessed": str(date.today()), "publisher": "D2L.ai Turkish translation community", "repository": "d2l-ai/d2l-tr",
                        "commit": COMMITS["tr"], "path": "TERMINOLOGY.md", "notes": "Direct English–Turkish pairs; some entries offer alternatives or require sense filtering."})
    if locale == "de":
        sources.append({"id": "kit-de-bilingual", "title": "Dictionary English–German, Appendix A.2 in KIT bachelor thesis",
                        "locale": "de", "kind": "academic bilingual technical glossary",
                        "url": "https://publish.etp.kit.edu/record/22293/files/ETP-Bachelor-KA-2025-01-Zeh-Marschke.pdf",
                        "accessed": str(date.today()), "publisher": "Karlsruhe Institute of Technology",
                        "notes": "Appendix A.2, PDF pages 53–54; selected pairs checked against a rendered page. Some less standard renderings are excluded."})
    sources.append({"id": f"amidi-{locale}-teaching", "title": f"CS 229/230 teaching cheatsheets, {locale} community translation",
                    "locale": locale, "kind": "bilingual teaching sheets",
                    "url": f"https://github.com/shervinea/cheatsheet-translation/tree/{CHEAT_COMMIT}/{locale}",
                    "accessed": str(date.today()), "publisher": "Shervine and Afshine Amidi translation community",
                    "repository": "shervinea/cheatsheet-translation", "commit": CHEAT_COMMIT, "license": "MIT",
                    "notes": "English numbered concept and translated counterpart appear in the same Markdown file. Community translation, not Stanford University publication."})
    sources += [{
        "id": f"google-ml-{locale}-{topic}", "title": f"Machine Learning Glossary: {topic} ({locale})",
        "locale": locale, "kind": "localized teaching glossary", "url": f"https://developers.google.com/machine-learning/glossary/{topic}?hl={locale}",
        "accessed": str(date.today()), "publisher": "Google for Developers",
        "notes": "The HTML lang attribute includes x-mtfrom-en, indicating machine translation from English; inspect terminology and definition.",
    } for topic in TOPICS]
    if locale == "ar":
        sources += [{
            "id": f"taima-ar-part{part}", "title": f"Dive into Deep Learning, Arabic translation by Dr. Alaa Taima, part {part}",
            "locale": "ar", "kind": "independent translated textbook PDF",
            "url": f"https://drive.google.com/file/d/{TAIMA_PDF_IDS[part]}/view",
            "accessed": str(date.today()), "publisher": "Dr. Alaa Taima / DL Arabic",
            "notes": "Independent edition; reuse license not established. Extracted PDF text has bidirectional/presentation-form artifacts; verify on rendered page.",
        } for part in (1, 2, 3)]
        pdf_pages = {part: [unicodedata.normalize("NFKC", "".join(ch for ch in page if unicodedata.category(ch) != "Cf"))
                            for page in (CORPORA / f"d2l-ar-taima-part{part}.txt").read_text(errors="replace").split("\f")]
                     for part in (1, 2, 3)}
    result = []
    for c in concepts:
        cid = c["id"]
        candidates = []
        for alias in [c["en"], *c.get("aliases_en", [])]:
            alias_key = key(alias)
            if locale == "de" and cid not in KIT_DENY:
                candidates.extend(("kit", hit) for hit in kk.get(alias_key, []))
            if cid not in CHEAT_DENY.get(locale, set()):
                candidates.extend(("cheat", hit) for hit in cc.get(alias_key, []))
            if locale == "tr" and cid not in TERMINOLOGY_DENY:
                # The same English 'bias' has explicitly labeled neural-network
                # and algorithmic senses; the concept ID chooses the right row.
                term_key = "bias" if cid in {"bias", "statistical-bias"} else alias_key
                for hit in tt.get(term_key, []):
                    if cid == "bias" and "sinir ağlarında" not in hit["term"]:
                        continue
                    if cid == "statistical-bias" and "algoritmalardaki" not in hit["term"]:
                        continue
                    candidates.append(("terminology", hit))
                if not candidates and alias_key.endswith("s"):
                    candidates.extend(("terminology", hit) for hit in tt.get(alias_key[:-1], []))
            if alias_key in dd and cid not in D2L_DENY.get(locale, set()):
                for hit in dd[alias_key]:
                    if hit["kind"] == "emphasis" and (hit["term"].count("(") > 1 or "translation" in hit["term"].casefold()):
                        continue
                    candidates.append(("d2l", hit))
            if alias_key in gd and cid not in GOOGLE_DENY.get(locale, set()):
                for hit in gd[alias_key]:
                    candidates.append(("google", hit))
            if candidates:
                break
        # Direct D2L headings are strongest; Google definitions beat a single
        # emphasized mention, especially for German's 2020 machine translation.
        priority = (lambda item: (0 if item[0] == "kit" else (1 if item[0] == "cheat" else (2 if item[0] == "google" else (3 if item[1]["kind"] == "heading" else 4))))) if locale == "de" else (lambda item: (0 if item[0] == "terminology" else (1 if item[0] == "d2l" and item[1]["kind"] == "heading" else (2 if item[0] == "cheat" else (3 if item[0] == "google" else 4)))))
        candidates.sort(key=priority)
        row = {"concept_id": cid, "preferred": None, "alternatives": [], "status": "unresolved", "source_ids": [], "evidence": [], "notes": "No exact aligned earlier-D2L or localized glossary head found; targeted source search needed."}
        if candidates:
            kind, hit = candidates[0]
            term = hit["term"].strip()
            if term:
                if kind == "cheat":
                    sid = f"amidi-{locale}-teaching"
                    locator = f"https://github.com/shervinea/cheatsheet-translation/blob/{CHEAT_COMMIT}/{hit['path']}#L{hit['line']}"
                    row.update(preferred=term, status="verified_external", source_ids=[sid],
                               evidence=[{"source_id": sid, "locator": locator, "source_term": hit["source_term"],
                                          "context": f"Numbered English explanation of ‘{c['en']}’ and its translated counterpart are adjacent in the same teaching sheet."}],
                               notes="Inspected bilingual teaching sheet; community translation, with preferred phrase normalized from parenthetical English where necessary.")
                elif kind == "kit":
                    sid = "kit-de-bilingual"
                    locator = f"https://publish.etp.kit.edu/record/22293/files/ETP-Bachelor-KA-2025-01-Zeh-Marschke.pdf#page={hit['page']}"
                    row.update(preferred=term, status="verified_external", source_ids=[sid],
                               evidence=[{"source_id": sid, "locator": locator, "source_term": term,
                                          "context": f"Appendix A.2 explicitly pairs English ‘{c['en']}’ with this German expression in an ML glossary."}],
                               notes="Academic English–German glossary pair; the technical sense was checked.")
                elif kind == "google":
                    sid = f"google-ml-{locale}-{hit['topic']}"
                    locator = f"https://developers.google.com/machine-learning/glossary/{hit['topic']}?hl={locale}#{hit['anchor']}"
                    context = f"Localized head for English ‘{c['en']}’; definition begins: {hit['target_context'][:130]}"
                    row.update(preferred=term, status="verified_external", source_ids=[sid], evidence=[{"source_id": sid, "locator": locator, "source_term": term, "context": context}], notes="Google page discloses machine translation (HTML lang x-mtfrom-en); target definition and technical sense require spot review.")
                else:
                    sid = "d2l-tr-terminology" if kind == "terminology" else f"d2l-{locale}-source"
                    locator = f"https://github.com/{REPOSITORIES[locale]}/blob/{COMMITS[locale]}/{hit['path']}#L{hit['line']}"
                    context = (f"Explicit editorial pair for English ‘{c['en']}’ in the D2L Turkish terminology list." if kind == "terminology" else f"Paired {hit['kind']} against English _origin.md for ‘{c['en']}’ in the same chapter.")
                    if kind == "terminology":
                        term = TERMINOLOGY_PREFERRED.get(cid, term.split(" veya ")[0].split(", ")[0].strip())
                    normalized = PREFERRED_NORMALIZE.get(locale, {}).get(cid)
                    if normalized:
                        term = normalized
                    row.update(preferred=term, status="verified_d2l", source_ids=[sid], evidence=[{"source_id": sid, "locator": locator, "source_term": hit['term'], "context": context}], notes="Reviewed explicit bilingual source pair." if kind == "terminology" else ("Preferred is the dictionary form of the inflected source expression; aligned source passage reviewed." if normalized else "Aligned D2L source expression and concept sense reviewed."))
        if locale == "tr" and cid == "dropout":
            # The official D2L editorial list uses the literal "hattan düşme".
            # Amidi's bilingual neural-net explanation uses the more idiomatic
            # technical label "Seyreltme" for the same random masking method.
            row.update(
                preferred="seyreltme", alternatives=["hattan düşme"], status="verified_external",
                source_ids=["amidi-tr-teaching", "d2l-tr-terminology"],
                evidence=[
                    {"source_id": "amidi-tr-teaching", "locator": f"https://github.com/shervinea/cheatsheet-translation/blob/{CHEAT_COMMIT}/tr/cs-230-deep-learning-tips-and-tricks.md#L324", "source_term": "Seyreltme", "context": "Adjacent English Dropout definition and Turkish neural-net description both specify random removal of units."},
                    {"source_id": "d2l-tr-terminology", "locator": f"https://github.com/d2l-ai/d2l-tr/blob/{COMMITS['tr']}/TERMINOLOGY.md#L102", "source_term": "hattan düşme", "context": "Official D2L editorial alternative for the same method."},
                ],
                notes="Preferred from an independent bilingual neural-network teaching sheet; D2L's explicit but literal alternative retained."
            )
        if locale == "ar" and cid in AR_PDF_APPROVED:
            preferred, part, pageno, source_term = AR_PDF_APPROVED[cid]
            page = pdf_pages[part][pageno - 1]
            if source_term not in page:
                print(f"PDF source term missing: {cid} part{part} page{pageno} {source_term!r}")
            else:
                sid = f"taima-ar-part{part}"
                source = next(src for src in sources if src["id"] == sid)
                locator = source["url"] + f"#page={pageno}"
                row.update(preferred=preferred, status="verified_d2l", source_ids=[sid],
                           evidence=[{"source_id": sid, "locator": locator, "source_term": source_term,
                                      "context": f"Arabic technical expression in Taima's D2L translation of {c['en']} (volume {part}, PDF page {pageno}); source text inspected."}],
                           notes="Preferred spelling/number normalized from the PDF's bidirectional text layer where needed; independent translation license not established.")
        if row["status"] == "unresolved" and cid in editorial and editorial[cid].get(locale):
            term = editorial[cid][locale]
            row.update(preferred=term, status="proposed", notes="Editorial candidate derived from the concept sense; needs same-sense publication review before automatic use.")
            occurrence = find_prose_occurrence(term, prose)
            if occurrence:
                kind, path, line_no, line = occurrence
                sid = f"d2l-{locale}-source" if kind == "d2l" else f"amidi-{locale}-teaching"
                repo = REPOSITORIES[locale] if kind == "d2l" else "shervinea/cheatsheet-translation"
                commit = COMMITS[locale] if kind == "d2l" else CHEAT_COMMIT
                locator = f"https://github.com/{repo}/blob/{commit}/{path}#L{line_no}"
                row["source_ids"] = [sid]
                row["evidence"] = [{"source_id": sid, "locator": locator, "source_term": term,
                                    "context": "This expression occurs in translated teaching prose; exact sense alignment still needs editorial review."}]
                row["notes"] = "Editorial candidate occurs in a translated source; remains proposed pending same-sense context review."
        result.append(row)
    return result, sources


def main() -> None:
    concepts = read_jsonl(BASE / "concepts.jsonl")
    sources = []
    for locale in LOCALES:
        rows, src = rows_for(locale, concepts)
        sources.extend(src)
        write_jsonl(BASE / "terms" / f"{locale}.jsonl", rows)
        from collections import Counter
        print(locale, len(rows), dict(Counter(r["status"] for r in rows)))
    write_jsonl(BASE / "sources" / "group-ar-tr-de.jsonl", sources)


if __name__ == "__main__":
    main()
