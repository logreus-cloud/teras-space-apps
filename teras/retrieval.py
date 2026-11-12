import math
import re
from collections import Counter

from .data import full_text

STOP_WORDS = frozenset(
    "a an and are as at be by for from has have in is it of on or that the this to was were with".split()
)


def _tokens(text):
    return [word for word in re.findall(r"[a-z0-9]+", text.lower()) if word not in STOP_WORDS]


class SearchIndex:
    def __init__(self, items, idf, vectors, norms):
        self.items = items
        self.idf = idf
        self.vectors = vectors
        self.norms = norms

    @classmethod
    def build(cls, items):
        items = tuple(items)
        documents = []
        for inv in items:
            boosted = inv.title + " " + " ".join(inv.keywords)
            documents.append(_tokens(full_text(inv)) + _tokens(boosted) * 2)
        frequencies = Counter(token for doc in documents for token in set(doc))
        idf = {
            token: math.log((len(items) + 1) / (count + 1)) + 1
            for token, count in frequencies.items()
        }
        vectors = tuple(
            {token: count * idf[token] for token, count in Counter(doc).items()}
            for doc in documents
        )
        norms = tuple(math.sqrt(sum(weight * weight for weight in vector.values())) for vector in vectors)
        return cls(items, idf, vectors, norms)

    def search(self, query, top_k=5):
        counts = Counter(_tokens(query))
        if not counts:
            return []
        query_vector = {token: count * self.idf.get(token, 1) for token, count in counts.items()}
        query_norm = math.sqrt(sum(weight * weight for weight in query_vector.values()))
        results = []
        for inv, vector, norm in zip(self.items, self.vectors, self.norms):
            score = sum(weight * vector.get(token, 0) for token, weight in query_vector.items())
            if score and norm:
                results.append((inv, score / (query_norm * norm)))
        results.sort(key=lambda result: (-result[1], result[0].accession))
        return results[:max(0, top_k)]
