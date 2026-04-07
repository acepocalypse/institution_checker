import asyncio

from institution_checker.main import (
    SURVEY_BORDERLINE,
    SURVEY_HARD_NO,
    SURVEY_PLAUSIBLE,
    SURVEY_STAGE_PASS_STRONG,
    SURVEY_STAGE_REJECT_FAST,
    _run_staged_pre_llm_search,
    _should_demote_validation_borderline_to_hard_no,
    classify_validation_borderline,
    evaluate_pre_llm_survey,
    run_pre_llm_survey,
    should_attempt_validation_enhanced,
    should_attempt_validation_rescue,
    should_skip_llm,
)
from institution_checker.llm_processor import (
    _auto_rescue_decision,
    _classify_evidence_window,
    _summarise_results,
    analyze_connection,
)
from institution_checker.search import _compute_signals
from institution_checker.search import ValidationSearchContext, validation_search_query


def make_result(
    *,
    title: str,
    snippet: str,
    url: str,
    relevance_score: int,
    has_person: bool = True,
    has_institution: bool = True,
    has_academic_role: bool = False,
    has_explicit_connection: bool = False,
    has_event_prize_pattern: bool = False,
):
    return {
        "title": title,
        "snippet": snippet,
        "url": url,
        "signals": {
            "has_person_name": has_person,
            "has_institution": has_institution,
            "has_academic_role": has_academic_role,
            "has_explicit_connection": has_explicit_connection,
            "has_event_prize_pattern": has_event_prize_pattern,
            "relevance_score": relevance_score,
        },
    }


def test_zero_results_are_hard_no():
    decision = evaluate_pre_llm_survey([], name="Nobody")
    assert decision.bucket == SURVEY_HARD_NO
    assert "no_results" in decision.reason_codes


def test_compute_signals_detects_historical_education_anchor():
    signals = _compute_signals(
        "Ben Example biography",
        "The U.S. Navy sent Ben Example to Purdue University for officers training and he remained there to receive a B.S. degree in 1947.",
        "https://history.example.org/ben-example",
        "Purdue University",
        "Ben Example",
    )
    assert signals["has_historical_education_anchor"] is True
    assert signals["has_explicit_connection"] is True


def test_compute_signals_detects_studied_under_historical_anchor():
    signals = _compute_signals(
        "Prize page",
        "Both scientists studied at Purdue University under Professor Herbert C. Brown before returning to Japan.",
        "https://www.chem.purdue.edu/negishi/research.php",
        "Purdue University",
        "Akira Example",
    )
    assert signals["has_historical_education_anchor"] is True


def test_compute_signals_rejects_nobel_cooccurrence_without_anchor():
    signals = _compute_signals(
        "Prize article",
        "Akira Example shared the Nobel Prize with a Purdue University professor.",
        "https://news.example.org/prize-story",
        "Purdue University",
        "Akira Example",
    )
    assert signals["has_historical_role_anchor"] is False
    assert signals["has_historical_education_anchor"] is False


def test_conference_only_results_are_hard_no():
    results = [
        make_result(
            title="Conference keynote by Alice Example",
            snippet="Alice Example will speak at a Purdue University workshop.",
            url="https://events.example.com/keynote",
            relevance_score=3,
            has_person=True,
            has_institution=False,
            has_event_prize_pattern=True,
        )
    ]

    decision = evaluate_pre_llm_survey(results, name="Alice Example")
    assert decision.bucket == SURVEY_HARD_NO
    assert should_skip_llm(results, name="Alice Example")[0] is True


def test_authoritative_purdue_profile_is_plausible():
    results = [
        make_result(
            title="Jane Example - Purdue University Directory",
            snippet="Jane Example is an assistant professor in the Department of Chemistry.",
            url="https://www.purdue.edu/directory/people/jane-example",
            relevance_score=16,
            has_person=True,
            has_institution=True,
            has_academic_role=True,
        )
    ]

    decision = evaluate_pre_llm_survey(results, name="Jane Example")
    assert decision.bucket == SURVEY_PLAUSIBLE
    assert should_skip_llm(results, name="Jane Example")[0] is False


def test_borderline_single_hit_stays_out_of_hard_no_before_rescue():
    results = [
        make_result(
            title="Mass spectrometry seminar at Purdue",
            snippet="John B. Fenn will speak at Purdue University about mass spectrometry.",
            url="https://www.purdue.edu/newsroom/releases/2004/q4/seminar.html",
            relevance_score=4,
            has_person=True,
            has_institution=True,
            has_event_prize_pattern=True,
        )
    ]

    decision = evaluate_pre_llm_survey(results, name="John B. Fenn")
    assert decision.bucket == SURVEY_BORDERLINE


def test_joint_campus_false_positive_is_hard_no():
    results = [
        make_result(
            title="IUPUI seminar series welcomes Alice Example",
            snippet="Indiana University-Purdue University Indianapolis hosts Alice Example.",
            url="https://iupui.edu/events/alice-example",
            relevance_score=4,
            has_person=True,
            has_institution=True,
            has_event_prize_pattern=True,
        )
    ]

    decision = evaluate_pre_llm_survey(results, name="Alice Example")
    assert decision.bucket == SURVEY_HARD_NO


def test_manual_positive_regressions_never_land_in_hard_no():
    fixtures = {
        "John B. Fenn": [
            make_result(
                title="John B. Fenn | Nobel Prize, Chemistry, Mass Spectrometry",
                snippet="John B. Fenn visited Purdue University and collaborated with faculty there.",
                url="https://www.britannica.com/biography/John-Fenn",
                relevance_score=11,
            ),
            make_result(
                title="Purdue chemists give an old laboratory bloodhound a sharper nose",
                snippet="John B. Fenn, who shared the Nobel Prize for chemistry, said the Purdue technique could help.",
                url="https://www.purdue.edu/uns/html4ever/2004/041015.Chem.Bloodhound.html",
                relevance_score=8,
            ),
        ],
        "Vernon L. Smith": [
            make_result(
                title="Vernon L. Smith Purdue economics lecture",
                snippet="Vernon L. Smith studied at Purdue University before his later faculty appointments.",
                url="https://economics.example.com/vernon-smith",
                relevance_score=10,
            )
        ],
        "Julian Schwinger": [
            make_result(
                title="Julian Schwinger at Purdue University",
                snippet="Julian Schwinger held a visiting appointment at Purdue University.",
                url="https://archives.example.org/julian-schwinger",
                relevance_score=10,
                has_explicit_connection=True,
            )
        ],
        "Wolfgang Pauli": [
            make_result(
                title="Wolfgang Pauli Purdue University visiting scholar notes",
                snippet="Wolfgang Pauli served as a visiting professor at Purdue University.",
                url="https://history.example.org/pauli-purdue",
                relevance_score=10,
                has_explicit_connection=True,
            )
        ],
    }

    for name, results in fixtures.items():
        decision = evaluate_pre_llm_survey(results, name=name)
        assert decision.bucket != SURVEY_HARD_NO, name


def test_rescue_query_can_promote_borderline(monkeypatch):
    initial_results = [
        make_result(
            title="Historical profile entry",
            snippet="John B. Fenn was sent to Purdue University for training and later remained there to receive a degree.",
            url="https://history.example.org/john-fenn-biography",
            relevance_score=8,
            has_person=True,
            has_institution=True,
            has_explicit_connection=True,
        )
    ]

    async def fake_validation_search_query(*args, **kwargs):
        return [
            make_result(
                title="John B. Fenn - Purdue University Faculty Profile",
                snippet="John B. Fenn served as a visiting professor at Purdue University.",
                url="https://www.purdue.edu/faculty/john-fenn",
                relevance_score=15,
                has_person=True,
                has_institution=True,
                has_academic_role=True,
                has_explicit_connection=True,
            )
        ], {"backend_used": "ddg", "cache_hit": False, "network_queries_used": 1}

    monkeypatch.setattr("institution_checker.main.validation_search_query", fake_validation_search_query)
    merged_results, decision = asyncio.run(run_pre_llm_survey("John B. Fenn", initial_results))

    assert len(merged_results) == 2
    assert decision.bucket == SURVEY_PLAUSIBLE
    assert decision.used_rescue_query is True
    assert "rescue_query_promoted" in decision.reason_codes


def test_rescue_query_keeps_anchored_borderline_out_of_hard_no(monkeypatch):
    initial_results = [
        make_result(
            title="Historical profile entry",
            snippet="John B. Fenn was sent to Purdue University for training and later remained there to receive a degree.",
            url="https://history.example.org/john-fenn-biography",
            relevance_score=8,
            has_person=True,
            has_institution=True,
            has_explicit_connection=True,
        )
    ]

    async def fake_validation_search_query(*args, **kwargs):
        return [], {"backend_used": "ddg", "cache_hit": False, "network_queries_used": 1}

    monkeypatch.setattr("institution_checker.main.validation_search_query", fake_validation_search_query)
    merged_results, decision = asyncio.run(run_pre_llm_survey("John B. Fenn", initial_results))

    assert len(merged_results) == 1
    assert decision.bucket == SURVEY_BORDERLINE
    assert decision.used_rescue_query is True
    assert "rescue_query_failed" in decision.reason_codes


def test_low_connection_strict_gate_rejects_weak_non_domain_evidence():
    results = [
        make_result(
            title="John Example won a conference award",
            snippet="John Example gave a keynote at Purdue University and was recognized.",
            url="https://events.example.com/john-example",
            relevance_score=7,
            has_person=True,
            has_institution=True,
            has_event_prize_pattern=True,
        )
    ]

    decision = evaluate_pre_llm_survey(results, dataset_profile="low_connection", name="John Example")
    assert decision.bucket == SURVEY_HARD_NO
    assert decision.metrics.get("stage") == SURVEY_STAGE_REJECT_FAST


def test_low_connection_strong_domain_evidence_still_passes():
    results = [
        make_result(
            title="John Example - Purdue University Directory",
            snippet="John Example is listed in the Purdue University faculty directory.",
            url="https://www.purdue.edu/directory/people/john-example",
            relevance_score=12,
            has_person=True,
            has_institution=True,
            has_academic_role=True,
        )
    ]

    decision = evaluate_pre_llm_survey(results, dataset_profile="low_connection", name="John Example")
    assert decision.bucket == SURVEY_PLAUSIBLE
    assert decision.metrics.get("stage") == SURVEY_STAGE_PASS_STRONG


def test_generic_purdue_mentions_do_not_make_celebrity_plausible():
    results = [
        make_result(
            title="Taylor Swift visits Purdue University for event",
            snippet="Taylor Swift spoke at Purdue University during a campus event.",
            url="https://www.purdue.edu/newsroom/taylor-swift-event",
            relevance_score=11,
            has_person=True,
            has_institution=True,
            has_event_prize_pattern=True,
        ),
        make_result(
            title="Purdue student newspaper mentions Taylor Swift",
            snippet="Students discussed Taylor Swift after a Purdue University concert watch party.",
            url="https://www.purdueexponent.org/arts/taylor-swift",
            relevance_score=9,
            has_person=True,
            has_institution=True,
        ),
    ]

    decision = evaluate_pre_llm_survey(results, name="Taylor Swift")
    assert decision.bucket != SURVEY_PLAUSIBLE


def test_authoritative_page_without_connection_shape_stays_out_of_plausible():
    results = [
        make_result(
            title="Lionel Messi featured by Purdue University news",
            snippet="Purdue University highlighted Lionel Messi in a sports analytics article.",
            url="https://www.purdue.edu/newsroom/messi-analytics-story",
            relevance_score=12,
            has_person=True,
            has_institution=True,
        )
    ]

    decision = evaluate_pre_llm_survey(results, name="Lionel Messi")
    assert decision.bucket != SURVEY_PLAUSIBLE


def test_multiple_weak_mentions_without_role_evidence_stay_borderline():
    results = [
        make_result(
            title="Taylor Swift mentioned in Purdue campus article",
            snippet="Purdue University students discussed Taylor Swift after a listening event.",
            url="https://www.purdue.edu/newsroom/taylor-swift-campus-story",
            relevance_score=10,
            has_person=True,
            has_institution=True,
        ),
        make_result(
            title="Purdue Exponent pop culture column on Taylor Swift",
            snippet="Taylor Swift was referenced by Purdue students in a campus opinion column.",
            url="https://www.purdueexponent.org/arts-and-culture/taylor-swift-column",
            relevance_score=9,
            has_person=True,
            has_institution=True,
        ),
    ]

    decision = evaluate_pre_llm_survey(results, name="Taylor Swift")
    assert decision.bucket == SURVEY_BORDERLINE


def test_celebrity_news_style_hits_demote_to_hard_no_after_rescue_shape():
    results = [
        make_result(
            title="Shakira featured in Purdue University article",
            snippet="Purdue University highlighted Shakira in a campus analytics article.",
            url="https://www.purdue.edu/newsroom/shakira-analytics-story",
            relevance_score=18,
            has_person=True,
            has_institution=True,
        ),
        make_result(
            title="Purdue article mentions Shakira",
            snippet="Students referenced Shakira in a Purdue University pop culture column.",
            url="https://www.purdue.edu/stories/shakira-column",
            relevance_score=14,
            has_person=True,
            has_institution=True,
        ),
    ]

    decision = evaluate_pre_llm_survey(results, name="Shakira", used_rescue_query=True)
    assert decision.bucket == SURVEY_HARD_NO


def test_domain_only_person_overlap_after_rescue_is_hard_no():
    results = [
        make_result(
            title="Oprah Winfrey referenced by Purdue students",
            snippet="Purdue University students mentioned Oprah Winfrey in a campus article.",
            url="https://www.purdue.edu/news/oprah-campus-article",
            relevance_score=20,
            has_person=True,
            has_institution=True,
        ),
        make_result(
            title="Purdue news article about Oprah Winfrey",
            snippet="An article about Oprah Winfrey appeared on a Purdue University site.",
            url="https://www.purdue.edu/newsroom/oprah-story",
            relevance_score=16,
            has_person=True,
            has_institution=True,
        ),
    ]

    decision = evaluate_pre_llm_survey(results, name="Oprah Winfrey", used_rescue_query=True)
    assert decision.bucket == SURVEY_HARD_NO


def test_strong_profile_case_sets_all_signal_families():
    results = [
        make_result(
            title="Jane Example - Purdue University Faculty Directory",
            snippet="Jane Example is an associate professor at Purdue University and earned her PhD from Purdue University.",
            url="https://www.purdue.edu/directory/people/jane-example",
            relevance_score=18,
            has_person=True,
            has_institution=True,
            has_academic_role=True,
            has_explicit_connection=True,
        ),
        make_result(
            title="Jane Example research profile",
            snippet="Jane Example is a Purdue University professor in the department directory.",
            url="https://www.purdue.edu/faculty/jane-example",
            relevance_score=16,
            has_person=True,
            has_institution=True,
            has_academic_role=True,
            has_explicit_connection=True,
        ),
    ]

    decision = evaluate_pre_llm_survey(results, name="Jane Example")
    assert decision.bucket == SURVEY_PLAUSIBLE
    assert decision.metrics["has_direct_connection_family"] is True
    assert decision.metrics["has_authoritative_profile_family"] is True
    assert decision.metrics["has_corroborating_support_family"] is True


def test_rescue_query_with_only_domain_noise_stays_hard_no(monkeypatch):
    initial_results = [
        make_result(
            title="Zendaya campus mention",
            snippet="Zendaya was referenced in a Purdue University student article.",
            url="https://student.example.com/zendaya",
            relevance_score=6,
            has_person=True,
            has_institution=True,
        )
    ]

    async def fake_validation_search_query(*args, **kwargs):
        return [
            make_result(
                title="Purdue University article mentioning Zendaya",
                snippet="Purdue University students mentioned Zendaya in a campus feature story.",
                url="https://www.purdue.edu/newsroom/zendaya-feature",
                relevance_score=17,
                has_person=True,
                has_institution=True,
            )
        ], {"backend_used": "ddg", "cache_hit": False, "network_queries_used": 1}

    monkeypatch.setattr("institution_checker.main.validation_search_query", fake_validation_search_query)
    merged_results, decision = asyncio.run(run_pre_llm_survey("Zendaya", initial_results))

    assert len(merged_results) == 1
    assert decision.bucket == SURVEY_HARD_NO
    assert decision.used_rescue_query is False


def test_remaining_celebrity_media_leaks_no_longer_plausible():
    fixtures = {
        "Oprah Winfrey": [
            make_result(
                title="Past Speakers - Sinai Forum - Purdue University Northwest",
                snippet="Oprah Winfrey appeared in the Sinai Forum speaker series.",
                url="https://www.pnw.edu/sinai-forum/past-speakers/oprah-winfrey",
                relevance_score=18,
                has_person=True,
                has_institution=True,
            ),
            make_result(
                title="Oprah Winfrey catalog record - Purdue Libraries",
                snippet="Library catalog entry for a book about Oprah Winfrey.",
                url="https://library.purdue.edu/discovery/oprah-winfrey",
                relevance_score=14,
                has_person=True,
                has_institution=True,
            ),
        ],
        "Shakira": [
            make_result(
                title="Shakira mentioned in Purdue marketing campaign",
                snippet="Purdue students referenced Shakira in a brand studio campaign.",
                url="https://www.purdue.edu/brand-studio/shakira-campaign",
                relevance_score=17,
                has_person=True,
                has_institution=True,
            )
        ],
        "Rihanna": [
            make_result(
                title="Rihanna inspires Purdue fashion feature",
                snippet="Purdue University published a campus feature inspired by Rihanna.",
                url="https://www.purdue.edu/stories/rihanna-fashion-feature",
                relevance_score=16,
                has_person=True,
                has_institution=True,
            )
        ],
        "Taylor Swift": [
            make_result(
                title="Taylor Swift themed Purdue event",
                snippet="Purdue students hosted a Taylor Swift themed watch party on campus.",
                url="https://www.purdue.edu/events/taylor-swift-watch-party",
                relevance_score=16,
                has_person=True,
                has_institution=True,
            )
        ],
        "Tom Hanks": [
            make_result(
                title="Tom Hanks in presidential lecture article",
                snippet="Purdue covered a lecture series article about Tom Hanks.",
                url="https://www.purdue.edu/presidential-lecture/tom-hanks-article",
                relevance_score=15,
                has_person=True,
                has_institution=True,
            )
        ],
        "Pablo Picasso": [
            make_result(
                title="Pablo Picasso in Purdue libraries catalog",
                snippet="Purdue Libraries catalog listing for a Pablo Picasso volume.",
                url="https://library.purdue.edu/catalog/pablo-picasso",
                relevance_score=12,
                has_person=True,
                has_institution=True,
            )
        ],
    }

    for name, results in fixtures.items():
        decision = evaluate_pre_llm_survey(results, name=name, used_rescue_query=True)
        assert decision.bucket != SURVEY_PLAUSIBLE, name


def test_purdue_library_catalog_result_is_not_authoritative():
    results = [
        make_result(
            title="Tom Hanks catalog record - Purdue Libraries",
            snippet="Library discovery listing for Tom Hanks biography.",
            url="https://library.purdue.edu/discovery/tom-hanks-biography",
            relevance_score=13,
            has_person=True,
            has_institution=True,
        )
    ]

    decision = evaluate_pre_llm_survey(results, name="Tom Hanks")
    assert decision.bucket != SURVEY_PLAUSIBLE
    assert decision.metrics["authoritative_institution_hits"] == 0


def test_generic_directory_search_landing_page_is_not_authoritative():
    results = [
        make_result(
            title="Purdue Directory Search Results",
            snippet="Search results for Jane Example in the Purdue directory.",
            url="https://www.purdue.edu/directory/?search=Jane+Example",
            relevance_score=14,
            has_person=True,
            has_institution=True,
        )
    ]

    decision = evaluate_pre_llm_survey(results, name="Jane Example")
    assert decision.bucket != SURVEY_PLAUSIBLE
    assert decision.metrics["authoritative_institution_hits"] == 0


def test_weak_but_real_person_specific_profile_page_can_still_pass():
    results = [
        make_result(
            title="Samuel D. Allen - Purdue University",
            snippet="Samuel D. Allen profile page in Purdue records.",
            url="https://www.purdue.edu/alumni/people/samuel-d-allen",
            relevance_score=15,
            has_person=True,
            has_institution=True,
        )
    ]

    decision = evaluate_pre_llm_survey(results, name="Samuel D. Allen")
    assert decision.bucket == SURVEY_PLAUSIBLE
    assert decision.metrics["weak_profile_anchor_hits"] >= 1


def test_legacy_positive_shape_can_be_plausible_after_rescue():
    results = [
        make_result(
            title="John B. Fenn and Purdue history",
            snippet="John B. Fenn served as a visiting professor at Purdue University.",
            url="https://history.example.org/john-b-fenn-purdue",
            relevance_score=25,
            has_person=True,
            has_institution=True,
            has_academic_role=True,
            has_explicit_connection=True,
        ),
        make_result(
            title="Purdue chemistry archive mentions John B. Fenn",
            snippet="Purdue University archives mention John B. Fenn and mass spectrometry.",
            url="https://www.purdue.edu/archive/john-fenn",
            relevance_score=18,
            has_person=True,
            has_institution=True,
        ),
        make_result(
            title="Purdue remembrance of John B. Fenn",
            snippet="Purdue University reflects on John B. Fenn's historical connection.",
            url="https://www.purdue.edu/newsroom/john-fenn-remembrance",
            relevance_score=16,
            has_person=True,
            has_institution=True,
        ),
        make_result(
            title="John B. Fenn on Purdue department page",
            snippet="A Purdue University department page mentions John B. Fenn.",
            url="https://engineering.purdue.edu/department/john-fenn",
            relevance_score=15,
            has_person=True,
            has_institution=True,
        ),
        make_result(
            title="John B. Fenn Purdue University reference",
            snippet="Purdue University reference page for John B. Fenn.",
            url="https://chemistry.purdue.edu/john-fenn-reference",
            relevance_score=14,
            has_person=True,
            has_institution=True,
        ),
    ]

    decision = evaluate_pre_llm_survey(results, name="John B. Fenn", used_rescue_query=True)
    assert decision.bucket == SURVEY_PLAUSIBLE
    assert "legacy_positive_shape" in decision.reason_codes


def test_validation_escalation_skips_obvious_negative():
    results = [
        make_result(
            title="Celebrity mentioned in article",
            snippet="Taylor Swift was mentioned in a campus article.",
            url="https://www.purdue.edu/newsroom/taylor-swift-story",
            relevance_score=8,
            has_person=True,
            has_institution=True,
        )
    ]

    decision = evaluate_pre_llm_survey(results, name="Taylor Swift", used_rescue_query=True)
    escalate, reason, subtype = should_attempt_validation_enhanced("Taylor Swift", decision)
    assert escalate is False
    assert reason == "obvious_negative"
    assert subtype in {"hard_no", "borderline_low_signal"}


def test_validation_escalation_promotes_borderline():
    results = [
        make_result(
            title="Historical biography entry",
            snippet="Alex Example served as a visiting professor at Purdue University in 1942.",
            url="https://history.example.org/alex-example",
            relevance_score=10,
            has_person=True,
            has_institution=True,
            has_academic_role=True,
            has_explicit_connection=True,
        )
    ]

    decision = evaluate_pre_llm_survey(results, name="Alex Example")
    escalate, reason, subtype = should_attempt_validation_rescue("Alex Example", decision)
    assert escalate is True
    assert reason in {"borderline", "explicit_connection", "historical_anchor"}
    assert subtype == "borderline_likely_positive"


def test_validation_escalation_promotes_weak_profile_anchor():
    results = [
        make_result(
            title="Samuel D. Allen - Purdue University",
            snippet="Samuel D. Allen profile page in Purdue records.",
            url="https://www.purdue.edu/alumni/people/samuel-d-allen",
            relevance_score=15,
            has_person=True,
            has_institution=True,
        )
    ]

    decision = evaluate_pre_llm_survey(results, name="Samuel D. Allen")
    escalate, reason, subtype = should_attempt_validation_enhanced("Samuel D. Allen", decision)
    assert escalate is False
    assert reason == "already_plausible"
    assert subtype == "already_plausible"


def test_validation_escalation_promotes_explicit_connection_without_promotion():
    results = [
        make_result(
            title="Historical page mentioning John B. Fenn",
            snippet="John B. Fenn served as a visiting professor at Purdue University.",
            url="https://history.example.org/john-fenn",
            relevance_score=12,
            has_person=True,
            has_institution=True,
            has_academic_role=True,
            has_explicit_connection=True,
        )
    ]

    decision = evaluate_pre_llm_survey(results, name="John B. Fenn", used_rescue_query=True)
    escalate, reason, subtype = should_attempt_validation_enhanced("John B. Fenn", decision)
    assert escalate is True
    assert reason in {"borderline", "explicit_connection_without_promotion", "legacy_positive_shape", "historical_anchor", "explicit_connection"}
    assert subtype == "borderline_likely_positive"


def test_validation_rescue_gate_blocks_negative_shape_borderline():
    results = [
        make_result(
            title="Taylor Swift mentioned in Purdue event story",
            snippet="Taylor Swift was mentioned in a Purdue University campus article.",
            url="https://www.purdue.edu/stories/taylor-swift-campus-story",
            relevance_score=10,
            has_person=True,
            has_institution=True,
        ),
        make_result(
            title="Purdue article references Taylor Swift",
            snippet="Students referenced Taylor Swift in a Purdue University pop culture feature.",
            url="https://www.purdue.edu/newsroom/taylor-swift-feature",
            relevance_score=9,
            has_person=True,
            has_institution=True,
        ),
    ]

    decision = evaluate_pre_llm_survey(results, name="Taylor Swift")
    rescue, reason, subtype = should_attempt_validation_rescue("Taylor Swift", decision)
    assert decision.bucket == SURVEY_BORDERLINE
    assert rescue is False
    assert reason == "negative_shape_without_anchor"
    assert subtype == "borderline_low_signal"


def test_validation_rescue_gate_allows_explicit_connection_borderline():
    results = [
        make_result(
            title="Historical page mentioning John B. Fenn",
            snippet="John B. Fenn served as a visiting professor at Purdue University.",
            url="https://history.example.org/john-fenn",
            relevance_score=12,
            has_person=True,
            has_institution=True,
            has_academic_role=True,
            has_explicit_connection=True,
        )
    ]

    decision = evaluate_pre_llm_survey(results, name="John B. Fenn")
    rescue, reason, subtype = should_attempt_validation_rescue("John B. Fenn", decision)
    assert rescue is True
    assert subtype == "borderline_likely_positive"


def test_validation_enhanced_gate_blocks_negative_shape_post_rescue():
    results = [
        make_result(
            title="Rihanna campus feature at Purdue",
            snippet="Rihanna was referenced in a Purdue University feature story.",
            url="https://www.purdue.edu/stories/rihanna-feature",
            relevance_score=11,
            has_person=True,
            has_institution=True,
        ),
        make_result(
            title="Purdue article mentions Rihanna",
            snippet="Students mentioned Rihanna in a Purdue University article.",
            url="https://www.purdue.edu/newsroom/rihanna-article",
            relevance_score=10,
            has_person=True,
            has_institution=True,
        ),
    ]

    decision = evaluate_pre_llm_survey(results, name="Rihanna", used_rescue_query=True)
    enhanced, reason, subtype = should_attempt_validation_enhanced("Rihanna", decision)
    assert enhanced is False
    assert reason in {"negative_shape_post_rescue", "obvious_negative"}
    assert subtype in {"borderline_low_signal", "hard_no"}


def test_classify_validation_borderline_marks_historical_anchor():
    results = [
        make_result(
            title="Archive entry for Alex Example",
            snippet="Alex Example earned a B.S. degree from Purdue University after officers training.",
            url="https://history.example.org/alex-example-purdue",
            relevance_score=10,
            has_person=True,
            has_institution=True,
            has_explicit_connection=True,
        )
    ]

    decision = evaluate_pre_llm_survey(results, name="Alex Example")
    subtype, reason = classify_validation_borderline("Alex Example", decision)
    assert subtype == "borderline_likely_positive"
    assert reason in {"historical_anchor", "explicit_connection", "legacy_positive_shape"}


def test_validation_demotes_low_signal_negative_borderline_to_hard_no():
    results = [
        make_result(
            title="Taylor Swift campus mention",
            snippet="Taylor Swift was mentioned in a Purdue University campus article.",
            url="https://www.purdue.edu/stories/taylor-swift-campus-story",
            relevance_score=10,
            has_person=True,
            has_institution=True,
        )
    ]

    decision = evaluate_pre_llm_survey(results, name="Taylor Swift")
    subtype, _ = classify_validation_borderline("Taylor Swift", decision)
    assert subtype == "borderline_low_signal"
    assert _should_demote_validation_borderline_to_hard_no("Taylor Swift", decision, subtype) is True


def test_validation_does_not_demote_historical_anchor_borderline():
    results = [
        make_result(
            title="Historical biography entry",
            snippet="Alex Example was sent to Purdue University for officers training and remained there to receive a B.S. degree.",
            url="https://history.example.org/alex-example-purdue",
            relevance_score=11,
            has_person=True,
            has_institution=True,
            has_explicit_connection=True,
        )
    ]

    decision = evaluate_pre_llm_survey(results, name="Alex Example")
    subtype, _ = classify_validation_borderline("Alex Example", decision)
    assert _should_demote_validation_borderline_to_hard_no("Alex Example", decision, subtype) is False


def test_validation_search_query_uses_cache(monkeypatch):
    calls = {"ddg": 0}

    async def fake_ddg(query, institution, person_name, limit, debug=False):
        calls["ddg"] += 1
        return [
            make_result(
                title="Jane Example - Purdue University",
                snippet="Jane Example is in the Purdue University directory.",
                url="https://www.purdue.edu/directory/people/jane-example",
                relevance_score=12,
                has_person=True,
                has_institution=True,
            )
        ]

    monkeypatch.setattr("institution_checker.search._duckduckgo_search_ddgs_library", fake_ddg)
    context = ValidationSearchContext(cache_enabled=True, allow_bing_fallback=False)

    first_results, first_meta = asyncio.run(
        validation_search_query(
            'Jane Example Purdue University',
            institution="Purdue University",
            person_name="Jane Example",
            limit=5,
            context=context,
        )
    )
    second_results, second_meta = asyncio.run(
        validation_search_query(
            'Jane Example Purdue University',
            institution="Purdue University",
            person_name="Jane Example",
            limit=5,
            context=context,
        )
    )

    assert calls["ddg"] == 1
    assert len(first_results) == len(second_results) == 1
    assert first_meta["cache_hit"] is False
    assert second_meta["cache_hit"] is True


def test_validation_search_query_caches_empty_results(monkeypatch):
    calls = {"ddg": 0}

    async def fake_ddg(query, institution, person_name, limit, debug=False):
        calls["ddg"] += 1
        return []

    monkeypatch.setattr("institution_checker.search._duckduckgo_search_ddgs_library", fake_ddg)
    context = ValidationSearchContext(cache_enabled=True, allow_bing_fallback=False)

    first_results, first_meta = asyncio.run(
        validation_search_query(
            'Nobody Purdue University',
            institution="Purdue University",
            person_name="Nobody",
            limit=5,
            context=context,
        )
    )
    second_results, second_meta = asyncio.run(
        validation_search_query(
            'Nobody Purdue University',
            institution="Purdue University",
            person_name="Nobody",
            limit=5,
            context=context,
        )
    )

    assert calls["ddg"] == 1
    assert first_results == []
    assert second_results == []
    assert first_meta["cache_hit"] is False
    assert second_meta["cache_hit"] is True


def test_validation_search_query_fast_ddg_mode_skips_slow_fallback(monkeypatch):
    calls = {"ddg": 0, "manual": 0, "bing": 0}

    async def fake_ddg(query, institution, person_name, limit, debug=False):
        calls["ddg"] += 1
        return []

    async def fake_manual(query, institution, person_name, limit, debug=False):
        calls["manual"] += 1
        return [], False

    async def fake_bing(*args, **kwargs):
        calls["bing"] += 1
        return []

    monkeypatch.setattr("institution_checker.search._duckduckgo_search_ddgs_library", fake_ddg)
    monkeypatch.setattr("institution_checker.search._duckduckgo_search_manual_html", fake_manual)
    monkeypatch.setattr("institution_checker.search.bing_search", fake_bing)

    results, meta = asyncio.run(
        validation_search_query(
            "Jane Example Purdue University",
            institution="Purdue University",
            person_name="Jane Example",
            limit=5,
            context=ValidationSearchContext(cache_enabled=False, allow_bing_fallback=False, allow_slow_ddg_fallback=False),
            allow_bing_fallback=False,
            allow_slow_ddg_fallback=False,
        )
    )

    assert results == []
    assert calls == {"ddg": 1, "manual": 0, "bing": 0}
    assert meta["ddg_library_used"] is True
    assert meta["ddg_manual_retry_used"] is False
    assert meta["ddg_browser_fallback_used"] is False


def test_validation_search_query_can_use_bing_recovery(monkeypatch):
    calls = {"ddg": 0, "bing": 0}

    async def fake_ddg(query, institution, person_name, limit, debug=False):
        calls["ddg"] += 1
        return []

    async def fake_bing(*args, **kwargs):
        calls["bing"] += 1
        return [
            make_result(
                title="Jane Example - Purdue University",
                snippet="Jane Example served as faculty at Purdue University.",
                url="https://www.purdue.edu/directory/people/jane-example",
                relevance_score=14,
                has_person=True,
                has_institution=True,
                has_academic_role=True,
            )
        ]

    monkeypatch.setattr("institution_checker.search._duckduckgo_search_ddgs_library", fake_ddg)
    monkeypatch.setattr("institution_checker.search.bing_search", fake_bing)

    results, meta = asyncio.run(
        validation_search_query(
            "Jane Example Purdue University",
            institution="Purdue University",
            person_name="Jane Example",
            limit=5,
            context=ValidationSearchContext(cache_enabled=False, allow_bing_fallback=True, allow_slow_ddg_fallback=False),
            allow_bing_fallback=True,
            allow_slow_ddg_fallback=False,
        )
    )

    assert len(results) == 1
    assert calls == {"ddg": 1, "bing": 1}
    assert meta["bing_fallback_used"] is True


def test_staged_production_search_skips_rescue_for_negative_shape(monkeypatch):
    async def fake_basic_search(*args, **kwargs):
        return [
            make_result(
                title="Taylor Swift campus mention",
                snippet="Taylor Swift was mentioned in a Purdue University campus article.",
                url="https://www.purdue.edu/stories/taylor-swift-campus-story",
                relevance_score=10,
                has_person=True,
                has_institution=True,
            )
        ], {"backend_used": "ddg", "cache_hit": False, "network_queries_used": 1, "network_attempt_count": 1}

    async def fake_validation_search_query(*args, **kwargs):
        raise AssertionError("rescue should not run for low-signal negative-shape borderline")

    async def fake_enhanced_search(*args, **kwargs):
        raise AssertionError("enhanced should not run for low-signal negative-shape borderline")

    monkeypatch.setattr("institution_checker.main.validation_basic_search", fake_basic_search)
    monkeypatch.setattr("institution_checker.main.validation_search_query", fake_validation_search_query)
    monkeypatch.setattr("institution_checker.main.validation_enhanced_search", fake_enhanced_search)

    results, decision, metadata = asyncio.run(
        _run_staged_pre_llm_search(
            "Taylor Swift",
            dataset_profile="high_connection",
            allow_enhanced=True,
        )
    )

    assert len(results) == 1
    assert decision.bucket == SURVEY_HARD_NO
    assert metadata["rescue_attempted"] is False
    assert metadata["enhanced_escalated"] is False
    assert metadata["search_mode_used"] == "basic_only_demoted"


def test_staged_production_search_recovers_empty_basic_results(monkeypatch):
    calls = {"basic": 0}

    async def fake_basic_search(*args, **kwargs):
        calls["basic"] += 1
        if calls["basic"] == 1:
            return [], {
                "backend_used": "ddg",
                "cache_hit": False,
                "network_queries_used": 1,
                "network_attempt_count": 1,
                "ddg_manual_retry_used": False,
                "ddg_browser_fallback_used": False,
                "bing_fallback_used": False,
            }
        return [
            make_result(
                title="Julian Schwinger - Purdue University history",
                snippet="Julian Schwinger served as an instructor at Purdue University from 1941 to 1943.",
                url="https://www.physics.purdue.edu/about/history/julian-schwinger",
                relevance_score=18,
                has_person=True,
                has_institution=True,
                has_academic_role=True,
                has_explicit_connection=True,
            )
        ], {
            "backend_used": "ddg|bing",
            "cache_hit": False,
            "network_queries_used": 1,
            "network_attempt_count": 1,
            "ddg_manual_retry_used": False,
            "ddg_browser_fallback_used": False,
            "bing_fallback_used": True,
        }

    monkeypatch.setattr("institution_checker.main.validation_basic_search", fake_basic_search)
    monkeypatch.setattr(
        "institution_checker.main.validation_search_query",
        lambda *args, **kwargs: asyncio.sleep(0, result=([], {"backend_used": "", "cache_hit": False, "network_queries_used": 0, "network_attempt_count": 0})),
    )

    results, decision, metadata = asyncio.run(
        _run_staged_pre_llm_search(
            "Julian Schwinger",
            dataset_profile="high_connection",
            allow_enhanced=False,
        )
    )

    assert calls["basic"] == 2
    assert len(results) == 1
    assert decision.bucket != SURVEY_HARD_NO
    assert metadata["empty_result_recovery_attempted"] is True
    assert metadata["empty_result_recovery_succeeded"] is True
    assert metadata["network_queries_used"] >= 2
    assert metadata["bing_fallback_used"] is True


def test_staged_production_search_uses_enhanced_for_likely_positive(monkeypatch):
    async def fake_basic_search(*args, **kwargs):
        return [
            make_result(
                title="Historical biography entry",
                snippet="John B. Fenn was sent to Purdue University for training and later remained there to receive a degree.",
                url="https://history.example.org/john-fenn-biography",
                relevance_score=8,
                has_person=True,
                has_institution=True,
                has_explicit_connection=True,
            )
        ], {"backend_used": "ddg", "cache_hit": False, "network_queries_used": 1, "network_attempt_count": 1}

    async def fake_validation_search_query(*args, **kwargs):
        return [
            make_result(
                title="John B. Fenn biography",
                snippet="John B. Fenn collaborated with Purdue University researchers.",
                url="https://example.org/john-fenn-purdue",
                relevance_score=11,
                has_person=True,
                has_institution=True,
                has_explicit_connection=True,
            )
        ], {"backend_used": "ddg", "cache_hit": False, "network_queries_used": 1, "network_attempt_count": 1}

    async def fake_enhanced_search(*args, **kwargs):
        return [
            make_result(
                title="John B. Fenn - Purdue University Faculty Profile",
                snippet="John B. Fenn served as a visiting professor at Purdue University.",
                url="https://www.purdue.edu/faculty/john-fenn",
                relevance_score=18,
                has_person=True,
                has_institution=True,
                has_academic_role=True,
                has_explicit_connection=True,
            )
        ], {"backend_used": "ddg", "cache_hit": False, "network_queries_used": 2, "network_attempt_count": 2}

    monkeypatch.setattr("institution_checker.main.validation_basic_search", fake_basic_search)
    monkeypatch.setattr("institution_checker.main.validation_search_query", fake_validation_search_query)
    monkeypatch.setattr("institution_checker.main.validation_enhanced_search", fake_enhanced_search)

    results, decision, metadata = asyncio.run(
        _run_staged_pre_llm_search(
            "John B. Fenn",
            dataset_profile="high_connection",
            allow_enhanced=True,
        )
    )

    assert decision.bucket == SURVEY_PLAUSIBLE
    assert metadata["rescue_attempted"] is True
    assert metadata["enhanced_escalated"] is True
    assert metadata["search_mode_used"] == "basic_plus_rescue_plus_enhanced"
    assert metadata["network_queries_used"] == 4


def test_llm_evidence_window_classifies_visiting_and_faculty():
    assert _classify_evidence_window("he served as a visiting professor at purdue university") == "Visiting"
    assert _classify_evidence_window("she was professor at purdue university for many years") == "Faculty"
    assert _classify_evidence_window("he graduated from purdue university with a phd") == "Alumni"


def test_auto_rescue_decision_recovers_historical_alumni():
    results = [
        make_result(
            title="John B. Fenn and Purdue history",
            snippet="John B. Fenn earned his PhD from Purdue University and later became a leading chemist.",
            url="https://www.purdue.edu/history/john-fenn",
            relevance_score=18,
            has_person=True,
            has_institution=True,
        )
    ]

    rescued = _auto_rescue_decision(
        {"connected": "N", "verdict": "not_connected"},
        "John B. Fenn",
        "Purdue University",
        results,
        vip_mode=True,
    )

    assert rescued is not None
    assert rescued["verdict"] == "connected"
    assert rescued["relationship_type"] == "Alumni"


def test_auto_rescue_decision_recovers_visiting_role():
    results = [
        make_result(
            title="Wolfgang Pauli at Purdue",
            snippet="During this time Pauli also acted as a visiting professor in several American universities, including Purdue University.",
            url="https://www.physics.purdue.edu/history/wolfgang-pauli",
            relevance_score=17,
            has_person=True,
            has_institution=True,
        )
    ]

    rescued = _auto_rescue_decision(
        {"connected": "N", "verdict": "not_connected"},
        "Wolfgang Pauli",
        "Purdue University",
        results,
        vip_mode=True,
    )

    assert rescued is not None
    assert rescued["verdict"] == "connected"
    assert rescued["relationship_type"] == "Visiting"


def test_auto_rescue_decision_rejects_event_speaker_false_positive():
    results = [
        make_result(
            title="Celebrity lecture at Purdue",
            snippet="Taylor Swift was a keynote speaker at Purdue University for a campus event.",
            url="https://www.purdue.edu/events/taylor-swift-keynote",
            relevance_score=16,
            has_person=True,
            has_institution=True,
        )
    ]

    rescued = _auto_rescue_decision(
        {"connected": "N", "verdict": "not_connected"},
        "Taylor Swift",
        "Purdue University",
        results,
        vip_mode=False,
    )

    assert rescued is None


def test_summarise_results_prioritizes_direct_role_edu_evidence():
    results = [
        make_result(
            title="Generic article",
            snippet="John B. Fenn was mentioned alongside Purdue University.",
            url="https://example.org/john-fenn-mention",
            relevance_score=30,
            has_person=True,
            has_institution=True,
        ),
        make_result(
            title="Purdue historical profile",
            snippet="John B. Fenn served as a visiting professor at Purdue University.",
            url="https://www.purdue.edu/history/john-fenn",
            relevance_score=12,
            has_person=True,
            has_institution=True,
        ),
    ]

    summary = _summarise_results(results, limit=2, min_relevance_score=8)
    assert "Purdue historical profile" in summary


def test_analyze_connection_auto_rescues_after_llm_negative(monkeypatch):
    async def fake_call_llm(prompt, debug=False):
        return {
            "verdict": "not_connected",
            "relationship_type": "None",
            "relationship_timeframe": "unknown",
            "verification_detail": "No connection found.",
            "summary": "No connection found.",
            "primary_source": "",
            "confidence": "low",
            "verification_status": "needs_review",
            "temporal_context": "unknown",
        }

    monkeypatch.setattr("institution_checker.llm_processor._call_llm", fake_call_llm)

    results = [
        make_result(
            title="Wolfgang Pauli at Purdue",
            snippet="During this time Pauli also acted as a visiting professor in several American universities, including Purdue University.",
            url="https://www.physics.purdue.edu/history/wolfgang-pauli",
            relevance_score=17,
            has_person=True,
            has_institution=True,
        )
    ]

    decision = asyncio.run(
        analyze_connection(
            "Wolfgang Pauli",
            "Purdue University",
            results,
            vip_mode=True,
        )
    )

    assert decision["verdict"] == "connected"
    assert decision["relationship_type"] == "Visiting"


def test_search_signals_detect_pauli_style_visiting_role_despite_book_context():
    signals = _compute_signals(
        "Wolfgang Pauli - Physics Book",
        "After serving as a lecturer at the University of Hamburg, Pauli was appointed as Professor of Theoretical Physics in Zurich and soon progressed to the position of visiting professor at Princeton, University of Michigan, and Purdue University.",
        "https://www.physicsbook.gatech.edu/Wolfgang_Pauli",
        "Purdue University",
        "Wolfgang Pauli",
    )

    assert signals["has_explicit_connection"] is True
    assert signals["has_academic_role"] is True


def test_search_signals_do_not_promote_generic_book_context_without_role_window():
    signals = _compute_signals(
        "Taylor Swift - Biography Book",
        "A biography book discussing how Purdue University students reacted to Taylor Swift.",
        "https://example.com/taylor-swift-book",
        "Purdue University",
        "Taylor Swift",
    )

    assert signals["has_explicit_connection"] is False
