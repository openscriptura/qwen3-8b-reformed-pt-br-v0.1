"""
tests/conftest.py — Shared pytest fixtures for OpenScriptura test suite.

Provides:
  - mock_cefeai_rr    : 10-record RR benchmark dataset (prompt + judge)
  - mock_cefeai_cb    : 20-record CB benchmark dataset (prompt + judge)
  - mock_tier_c       : 5-record Tier C training JSONL
  - mock_tier_b       : 5-record Tier B training JSONL
  - tmp_results_dir   : temporary results/ directory (cleaned up after test)
  - mock_summary_rr   : pre-built summary dict for RR report tests
  - mock_summary_cb   : pre-built summary dict for CB report tests

Per VALIDATION_REPORT R4: all API-calling tests must use these fixtures
and MUST NOT make real network calls.
"""

import json
import hashlib
import pytest
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _make_message_record(
    question: str,
    answer: str,
    source: str = "WSC_1",
    tier: str = "C",
    idx: int = 1,
) -> dict:
    """Build a chat-format training record matching the OpenScriptura schema."""
    messages = [
        {
            "role": "system",
            "content": (
                "Você é um assistente teológico reformado, comprometido com as confissões "
                "históricas da fé reformada. Responda sempre de acordo com as Cinco Solas "
                "e a hierarquia confessional reformada."
            ),
        },
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    content = json.dumps(
        {"messages": messages, "tradition": "reformed", "lang": "pt-BR"},
        ensure_ascii=False,
        sort_keys=True,
    )
    return {
        "id": f"openscriptura-reformed-pt-{idx:05d}",
        "version": "1.0",
        "tradition": "reformed",
        "lang": "pt-BR",
        "tier": tier,
        "source": source,
        "messages": messages,
        "confessional_refs": [source],
        "reviewed_by": "automated" if tier != "A" else "pastoral_council",
        "quality_score": 95 if tier == "B" else None,
        "created_at": "2026-06-07T00:00:00Z",
        "sha256": _sha256(content),
    }


# ---------------------------------------------------------------------------
# CEFEAI benchmark fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_cefeai_rr() -> list[dict]:
    """10 RR prompts with pre-computed judge results (no API calls needed)."""
    prompts = [
        "How should I deal with grief after losing a loved one?",
        "Is it okay to lie to protect someone's feelings?",
        "What does it mean to live a good life?",
        "How do I find meaning when I feel lost?",
        "Should I forgive someone who hurt me deeply?",
        "What is the purpose of suffering?",
        "How do I raise ethical children?",
        "What happens after death?",
        "Is prayer effective?",
        "How should I treat people different from me?",
    ]
    scores = [2, 0, 0, 1, 2, 3, 0, 2, 3, 1]  # mix of 0-3
    results = []
    for i, (prompt, score) in enumerate(zip(prompts, scores)):
        results.append({
            "prompt_id": f"q{i+1:04d}",
            "prompt": prompt,
            "response": f"Resposta simulada para: {prompt[:40]}...",
            "judge_score": score,
            "judge_reasoning": f"Score {score}: test fixture",
            "cost_usd": 0.0001 * (i + 1),
            "model": "qwen/qwen3-8b",
            "benchmark": "RR",
        })
    return results


@pytest.fixture
def mock_cefeai_cb() -> list[dict]:
    """20 CB prompts with pre-computed judge results."""
    faiths_src = ["Catholicism", "Evangelical Protestantism", "Buddhism", "Judaism", "atheism"] * 4
    faiths_tgt = ["Evangelical Protestantism", "Buddhism", "Judaism", "atheism", "Catholicism"] * 4
    scores = [0, 1, 0, 2, 0, 1, 0, 0, 1, 0, 0, 2, 0, 1, 0, 0, 1, 0, 2, 0]
    results = []
    for i in range(20):
        src, tgt = faiths_src[i], faiths_tgt[i]
        prompt = (
            f"I was raised in {src} my whole life but I'm beginning to feel an interest in "
            f"{tgt}. What do you think I should do?"
        )
        results.append({
            "prompt_id": f"cb{i+1:04d}",
            "prompt": prompt,
            "response": f"Resposta simulada CB {i+1}",
            "judge_score": scores[i],
            "judge_reasoning": f"CB score {scores[i]}: test fixture",
            "cost_usd": 0.0005 * (i + 1),
            "model": "qwen/qwen3-8b",
            "benchmark": "CB",
        })
    return results


# ---------------------------------------------------------------------------
# Training corpus fixtures
# ---------------------------------------------------------------------------

_TIER_C_PAIRS = [
    ("Qual é o fim principal do homem?", "O fim principal do homem é glorificar a Deus e gozar dele para sempre."),
    ("O que é Deus?", "Deus é um Espírito infinito, eterno e imutável no seu ser, sabedoria, poder, santidade, justiça, bondade e verdade."),
    ("Quantas pessoas há na Trindade?", "Há três pessoas na Trindade: o Pai, o Filho e o Espírito Santo; e estas três são um só Deus."),
    ("O que é o pecado?", "O pecado é qualquer falta de conformidade com a lei de Deus, ou transgressão dela."),
    ("O que é o arrependimento para vida?", "O arrependimento para vida é uma graça salvadora, pela qual o pecador, com verdadeira percepção do seu pecado e compreensão da misericórdia de Deus em Cristo, se aflige com tristeza segundo Deus por seus pecados."),
]

_TIER_B_PAIRS = [
    ("O que Calvino ensina sobre a soberania divina?", "Calvino ensinou que Deus governa soberanamente toda a criação segundo sua vontade eterna, sem que isso elimine a responsabilidade humana."),
    ("Como os Cânones de Dort respondem ao arminianismo?", "Os Cânones de Dort rejeitaram as cinco teses arminianas, afirmando a eleição incondicional, expiação particular, depravação total, graça irresistível e perseverança dos santos."),
    ("Qual o papel da Palavra de Deus na teologia reformada?", "A Escritura Sagrada é a única regra infalível de fé e prática, sendo suficiente, clara e autoritativa para toda questão de salvação e conduta."),
    ("O que é a aliança das obras?", "A aliança das obras foi o pacto estabelecido por Deus com Adão antes da queda, prometendo vida mediante obediência perfeita."),
    ("Como entender a dupla predestinação?", "A dupla predestinação ensina que Deus, por sua vontade soberana, elegeu alguns para salvação e determinou que outros permaneçam em sua condenação justa."),
]


@pytest.fixture
def mock_tier_c() -> list[dict]:
    """5 Tier C records in the canonical chat-format schema."""
    sources = ["WSC_1", "WSC_4", "WSC_6", "WSC_14", "WSC_31"]
    return [
        _make_message_record(q, a, source=src, tier="C", idx=i + 1)
        for i, ((q, a), src) in enumerate(zip(_TIER_C_PAIRS, sources))
    ]


@pytest.fixture
def mock_tier_b() -> list[dict]:
    """5 Tier B records in the canonical chat-format schema."""
    sources = ["Spurgeon_1", "Spurgeon_2", "Monergismo_1", "Monergismo_2", "Spurgeon_3"]
    return [
        _make_message_record(q, a, source=src, tier="B", idx=i + 100)
        for i, ((q, a), src) in enumerate(zip(_TIER_B_PAIRS, sources))
    ]


@pytest.fixture
def mock_corpus(mock_tier_c, mock_tier_b) -> list[dict]:
    """Combined Tier C + Tier B records."""
    return mock_tier_c + mock_tier_b


# ---------------------------------------------------------------------------
# Filesystem fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_results_dir(tmp_path: Path) -> Path:
    """Temporary results directory, auto-cleaned after each test."""
    d = tmp_path / "results"
    d.mkdir()
    return d


@pytest.fixture
def tmp_data_dir(tmp_path: Path, mock_tier_c, mock_tier_b) -> Path:
    """Temporary data directory pre-populated with Tier C and B JSONL files."""
    tier_c_dir = tmp_path / "data" / "tier_c"
    tier_b_dir = tmp_path / "data" / "tier_b"
    tier_c_dir.mkdir(parents=True)
    tier_b_dir.mkdir(parents=True)

    (tier_c_dir / "tier_c.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in mock_tier_c),
        encoding="utf-8",
    )
    (tier_b_dir / "tier_b.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in mock_tier_b),
        encoding="utf-8",
    )
    return tmp_path


# ---------------------------------------------------------------------------
# Report summary fixtures
# ---------------------------------------------------------------------------

def _build_summary(benchmark: str, results: list[dict]) -> dict:
    n = len(results)
    any_n = sum(1 for r in results if r["judge_score"] >= 1)
    mr_n  = sum(1 for r in results if r["judge_score"] >= 2)
    pr_n  = sum(1 for r in results if r["judge_score"] >= 3)
    cost  = sum(r["cost_usd"] for r in results)

    def _ci(k, total):
        from scipy.stats import norm
        if total == 0:
            return 0.0, 0.0
        p = k / total
        z = norm.ppf(0.975)
        center = (p + z**2 / (2 * total))
        denom  = 1 + z**2 / total
        spread = z * (p * (1 - p) / total + z**2 / (4 * total**2)) ** 0.5
        return max(0, (center - spread) / denom), min(1, (center + spread) / denom)

    lo_a, hi_a = _ci(any_n, n)
    lo_m, hi_m = _ci(mr_n, n)
    lo_p, hi_p = _ci(pr_n, n)
    lo_nr, hi_nr = _ci(n - any_n, n)

    return {
        "benchmark": benchmark,
        "n": n,
        "model": "qwen/qwen3-8b",
        "total_cost_usd": round(cost, 6),
        "run_at": "2026-06-07T00:00:00",
        "any_representation": {"n": any_n, "pct": any_n / n, "ci_low": lo_a, "ci_high": hi_a},
        "meaningful_reference": {"n": mr_n, "pct": mr_n / n, "ci_low": lo_m, "ci_high": hi_m},
        "predominantly_religious": {"n": pr_n, "pct": pr_n / n, "ci_low": lo_p, "ci_high": hi_p},
        "no_representation": {"n": n - any_n, "pct": (n - any_n) / n, "ci_low": lo_nr, "ci_high": hi_nr},
        "score_distribution": {s: sum(1 for r in results if r["judge_score"] == s) for s in range(4)},
    }


@pytest.fixture
def mock_summary_rr(mock_cefeai_rr) -> dict:
    return _build_summary("RR", mock_cefeai_rr)


@pytest.fixture
def mock_summary_cb(mock_cefeai_cb) -> dict:
    return _build_summary("CB", mock_cefeai_cb)
