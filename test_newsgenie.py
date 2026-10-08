"""Tests for query routing, fallbacks and error handling.

Run with:  python -m pytest -q   (or: python -m unittest discover tests)

External APIs are mocked, so these tests need no keys or internet.
`run_workflow` mirrors the edges in graph.py so the routing logic can be
tested even without LangGraph installed.
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from newsgenie import classifier, config, credibility, news_api, nodes, web_search  # noqa: E402
from newsgenie.http_utils import APIError  # noqa: E402


def run_workflow(state):
    """Same routing as graph.py, without LangGraph."""
    state = dict(state)
    history = list(state.get("history", []))

    def apply(update):
        for k, v in update.items():
            if k == "history":
                history.extend(v)
            else:
                state[k] = v

    apply(nodes.start(state))
    step = nodes.route_after_start(state)
    while step != "finalize":
        apply(getattr(nodes, step)(state))
        if step == "classify":
            step = nodes.route_by_intent(state)
        elif step == "fetch_news":
            step = nodes.route_after_news(state)
        elif step == "search":
            step = nodes.route_after_search(state)
        else:
            step = "finalize"
    state["history"] = history
    apply(nodes.finalize(state))
    state["history"] = history
    return state


ARTICLE = {"title": "Markets rally on rate news", "description": "Stocks rose.",
           "url": "https://www.reuters.com/markets/x", "source": "Reuters",
           "published_at": "2026-10-08T12:00:00Z", "provider": "NewsAPI"}


def no_keys():
    return mock.patch.object(config, "get_key", return_value=None)


class ClassifierTests(unittest.TestCase):
    def test_news_requests(self):
        for q in ["latest AI news", "What's happening in sports today?",
                  "show me finance headlines", "any updates on Tesla"]:
            self.assertEqual(classifier.rule_based(q)["intent"], "news", q)

    def test_web_lookups(self):
        for q in ["What is the current price of bitcoin?", "weather in Detroit",
                  "who is the current CEO of Apple right now"]:
            self.assertEqual(classifier.rule_based(q)["intent"], "web", q)

    def test_general_questions(self):
        for q in ["Explain how photosynthesis works", "Write a haiku about autumn",
                  "What is a stock index?"]:
            self.assertEqual(classifier.rule_based(q)["intent"], "general", q)

    def test_category_detection(self):
        self.assertEqual(classifier.rule_based("latest NBA scores")["category"], "sports")
        self.assertEqual(classifier.rule_based("news about the stock market")["category"], "finance")
        self.assertEqual(classifier.rule_based("AI startup news")["category"], "technology")
        # word boundaries: "software" must not trigger "war" -> world
        self.assertEqual(classifier.rule_based("software news")["category"], "technology")

    def test_topic_extraction(self):
        self.assertEqual(classifier.rule_based("show me the latest news about Tesla")["topic"], "tesla")
        self.assertEqual(classifier.rule_based("latest tech news")["topic"], "")

    def test_follow_up_keeps_context(self):
        r = classifier.rule_based("tell me more about that", last_intent="news",
                                  last_topic="tesla")
        self.assertEqual((r["intent"], r["topic"]), ("news", "tesla"))


class CredibilityTests(unittest.TestCase):
    def test_trusted_beats_unknown_and_satire(self):
        trusted = credibility.score_article(ARTICLE)
        unknown = credibility.score_article({**ARTICLE, "url": "https://randomblog.biz/a"})
        satire = credibility.score_article({**ARTICLE, "url": "https://www.theonion.com/a"})
        self.assertEqual(trusted["credibility_label"], "High")
        self.assertGreater(trusted["credibility"], unknown["credibility"])
        self.assertEqual(satire["credibility_label"], "Low")

    def test_clickbait_penalty(self):
        bait = credibility.score_article({**ARTICLE, "title": "You won't believe what stocks did!!"})
        self.assertLess(bait["credibility"], credibility.score_article(ARTICLE)["credibility"])

    def test_filter_removes_junk_and_duplicates(self):
        items = [ARTICLE,
                 {**ARTICLE, "title": "Markets rally on rate news - CNBC", "url": "https://cnbc.com/x"},
                 {**ARTICLE, "title": "[Removed]"},
                 {**ARTICLE, "url": "https://theonion.com/z", "title": "Totally different"}]
        out = credibility.filter_and_rank(items)
        self.assertEqual(len(out), 1)
        self.assertIn("reuters", out[0]["url"])

    def test_trusted_only(self):
        items = [ARTICLE, {**ARTICLE, "title": "Other story", "url": "https://blog.example/x"}]
        self.assertEqual(len(credibility.filter_and_rank(items, trusted_only=True)), 1)


class NewsApiFallbackTests(unittest.TestCase):
    def setUp(self):
        news_api.clear_cache()

    def test_demo_mode_without_keys(self):
        with no_keys():
            r = news_api.fetch_news("technology")
        self.assertTrue(r["demo"])
        self.assertTrue(r["articles"])

    def test_falls_back_to_gnews_when_newsapi_fails(self):
        def bad(*a):
            raise APIError("NewsAPI", "server")
        with mock.patch.object(news_api, "PROVIDERS",
                               [("NewsAPI", bad), ("GNews", lambda c, q: [ARTICLE])]):
            r = news_api.fetch_news("finance")
        self.assertEqual(r["provider"], "GNews")
        self.assertIn("NewsAPI is having problems", r["errors"])

    def test_all_providers_fail_returns_empty_not_exception(self):
        def bad(*a):
            raise APIError("X", "timeout")
        with mock.patch.object(news_api, "PROVIDERS", [("A", bad), ("B", bad)]):
            r = news_api.fetch_news("sports")
        self.assertEqual(r["articles"], [])
        self.assertFalse(r["demo"])
        self.assertEqual(len(r["errors"]), 2)

    def test_cache(self):
        calls = []
        def provider(c, q):
            calls.append(1)
            return [ARTICLE]
        with mock.patch.object(news_api, "PROVIDERS", [("NewsAPI", provider)]):
            news_api.fetch_news("finance")
            r = news_api.fetch_news("finance")
        self.assertEqual(len(calls), 1)
        self.assertTrue(r["cached"])


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        news_api.clear_cache()

    def test_empty_input(self):
        out = run_workflow({"query": "   "})
        self.assertEqual(out["intent"], "invalid")
        self.assertIn("Please type", out["answer"])

    def test_headlines_button_demo_mode(self):
        with no_keys():
            out = run_workflow({"query": "", "mode": "headlines",
                                "selected_category": "sports"})
        self.assertEqual(out["intent"], "news")
        self.assertTrue(out["articles"])
        self.assertIn("Demo mode", out["answer"])
        self.assertIn("summarize_news:plain", out["trace"])

    def test_news_falls_back_to_web_search(self):
        empty = {"articles": [], "provider": "", "errors": ["NewsAPI timed out"],
                 "demo": False, "cached": False}
        found = {"results": [{"title": "Story", "url": "https://apnews.com/s",
                              "content": "Details", "source": "apnews.com"}],
                 "provider": "Tavily", "errors": []}
        with no_keys(), mock.patch.object(news_api, "fetch_news", return_value=empty), \
                mock.patch.object(web_search, "search_web", return_value=found):
            out = run_workflow({"query": "latest news on the mayor race"})
        self.assertTrue(out["used_fallback"])
        self.assertIn("web search", out["answer"])
        self.assertIn("NewsAPI timed out", out["errors"])
        self.assertEqual(out["trace"][-2], "answer_from_search")

    def test_nothing_found_anywhere(self):
        empty_news = {"articles": [], "provider": "", "errors": [], "demo": False, "cached": False}
        empty_web = {"results": [], "provider": "", "errors": []}
        with no_keys(), mock.patch.object(news_api, "fetch_news", return_value=empty_news), \
                mock.patch.object(web_search, "search_web", return_value=empty_web):
            out = run_workflow({"query": "breaking news about zzqx"})
        self.assertIn("couldn't find any live results", out["answer"])
        self.assertEqual(out["trace"][-2], "general_answer")

    def test_general_question_without_llm_key(self):
        empty_web = {"results": [], "provider": "", "errors": []}
        with no_keys(), mock.patch.object(web_search, "search_web", return_value=empty_web):
            out = run_workflow({"query": "Explain photosynthesis"})
        self.assertEqual(out["intent"], "general")
        self.assertIn("AI model is unavailable", out["answer"])

    def test_history_and_follow_up(self):
        with no_keys():
            first = run_workflow({"query": "latest technology news",
                                  "selected_category": "technology"})
            second = run_workflow({**first, "query": "tell me more about that"})
        self.assertEqual(len(second["history"]), 4)
        self.assertEqual(second["intent"], "news")

    def test_llm_used_when_available(self):
        def fake_ask(system, user, history=None, json_mode=False):
            if json_mode:  # the classifier call
                return '{"intent":"general","category":"general","topic":"","reason":"x"}', None
            return "An answer.", None

        with mock.patch.object(nodes.llm, "available", return_value=True), \
                mock.patch.object(nodes.llm, "ask", side_effect=fake_ask):
            out = run_workflow({"query": "what is inflation?"})
        self.assertEqual(out["classified_by"], "llm")
        self.assertEqual(out["answer"], "An answer.")

    def test_llm_bad_json_falls_back_to_rules(self):
        with mock.patch.object(classifier.llm, "available", return_value=True), \
                mock.patch.object(classifier.llm, "ask", return_value=("not json", None)):
            r = classifier.classify("latest sports news")
        self.assertEqual((r["intent"], r["method"]), ("news", "rules"))


class HttpTests(unittest.TestCase):
    def test_auth_error_not_retried(self):
        from newsgenie import http_utils
        resp = mock.Mock(status_code=401)
        with mock.patch.object(http_utils.requests, "request", return_value=resp) as req:
            with self.assertRaises(APIError) as ctx:
                http_utils.request_json("NewsAPI", "GET", "https://x")
        self.assertEqual(ctx.exception.kind, "auth")
        self.assertEqual(req.call_count, 1)

    def test_server_error_retried(self):
        from newsgenie import http_utils
        bad, good = mock.Mock(status_code=503), mock.Mock(status_code=200)
        good.json.return_value = {"ok": True}
        with mock.patch.object(http_utils.requests, "request", side_effect=[bad, good]), \
                mock.patch.object(http_utils.time, "sleep"):
            self.assertEqual(http_utils.request_json("S", "GET", "https://x"), {"ok": True})

    def test_placeholder_key_treated_as_missing(self):
        with mock.patch.dict(os.environ, {"NEWSAPI_KEY": "your_newsapi_key_here"}):
            self.assertIsNone(config.get_key("NEWSAPI_KEY"))


if __name__ == "__main__":
    unittest.main()
