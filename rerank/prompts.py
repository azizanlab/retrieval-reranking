"""Prompt templates for ranking and availability controls.

Candidates carry two-digit labels in a seeded presentation order. The pointwise
readout scores the answer string "Ranking: ii" for each candidate.
"""

SYSTEM = ("You are a shopping assistant. Given a customer's purchase history and what "
          "they are looking for, you rank candidate products from most to least likely "
          "to be the one they buy.")



HEADER_AVAILABILITY = "\n\n[Retrieved information for some candidates]\n"
HEADER_SELECTION = "\n\n[Retrieved reviews for the candidates]\n"


def user_prompt(history, query, candidates, cand_block=None):
    """history: titles, oldest first; candidates: [(label, title)]; cand_block replaces the plain
    candidate list (the metadata setting renders every candidate with its fields)."""
    n = len(candidates)
    cb = cand_block if cand_block is not None else "\n".join(f"[{lab}] {t}" for lab, t in candidates)
    parts = []
    if history:
        parts.append("Customer's recent purchase history (oldest to newest):\n" + "\n".join(f"- {t}" for t in history))
    if query:
        parts.append(f"What the customer is looking for:\n{query}")
    return ("\n\n".join(parts) + f"\n\nCandidate products to rank (indices 00-{n-1:02d}):\n{cb}"
            f"\n\nOutput a single final line:\n"
            f"`Ranking: i1,i2,...,i{n}` where each i is a two-digit index "
            f"(00-{n-1:02d}), most-likely first, including every index exactly once.")


def answer(i):
    """The answer string scored for candidate i."""
    return "Ranking: %02d" % i


def format_reviews(label, title, reviews, n_shown):
    lines = [f"[{label}] {title}"]
    for r in reviews[:n_shown]:
        rating = r.get("rating")
        star = f"{int(rating)}* " if rating is not None else ""
        lines.append(f'  - {star}"{r.get("text", "")}"')
    if not reviews:
        lines.append("  - (no reviews available)")
    return "\n".join(lines)



META_KEYS = ["price", "average_rating", "rating_number", "store", "categories", "features", "description",
             "year", "runtime", "genres", "director", "cast", "country"]


def format_metadata(label, meta):
    fields = []
    for key in META_KEYS:
        v = meta.get(key)
        if v in (None, "", [], 0, 0.0):
            continue
        if isinstance(v, list):
            v = ", ".join(map(str, v))
        fields.append(f"    {key}: {v}")
    return f"[{label}] {meta.get('title', '')}" + ("\n" + "\n".join(fields) if fields else "")


def retrieved_block(formatted_items, mode):
    what = "customer reviews" if mode == "reviews" else "product details"
    return f"[Retrieved {what}]\n" + "\n".join(formatted_items) + "\n[end]\n"



NO_INFO = "(no information available)"
_FILLER = ("this entry was last checked by the catalogue team and the record below was filed "
           "for internal use only no further details were attached to this listing at the time "
           "the note is retained here so that the layout of each entry stays the same across "
           "items and it carries nothing about the product itself").split()


def filler_words(n):
    return " ".join(_FILLER[i % len(_FILLER)] for i in range(max(1, n)))


def _head_pad(formatted):
    """The heading line of a formatted item, and the indent its detail lines use."""
    lines = formatted.splitlines()
    if len(lines) < 2:
        return lines[0], "  - "
    d = lines[1]
    return lines[0], d[:len(d) - len(d.lstrip())] + ("- " if d.lstrip().startswith("- ") else "")


def template_evidence(formatted):
    """Keep the heading, replace the content with a line saying there is nothing."""
    head, pad = _head_pad(formatted)
    return head + "\n" + pad + NO_INFO


def filler_evidence(formatted):
    """Keep the heading, replace the content with the same number of words of boilerplate (the
    pad's own bullet is charged once, so the word count matches exactly)."""
    lines = formatted.splitlines()
    head, pad = _head_pad(formatted)
    n = sum(len(l.split()) for l in lines[1:]) - len(pad.split())
    return head + "\n" + pad + filler_words(n)





THINK = {
    "qwen": {"open": "<think>\n", "close": "\n</think>\n\n", "empty": "<think>\n\n</think>\n\n"},
    "gpt-oss": {"open": "<|channel|>analysis<|message|>",
                "close": "<|end|><|start|>assistant<|channel|>final<|message|>",
                "empty": "<|channel|>final<|message|>"},
}
