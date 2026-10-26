"""Tests for the ingredient assistant: intents, mention finding, reference resolution, answers from the
knowledge base, refusals and the chat UI component.   Run:  pytest tests/test_assistant.py -q"""
import pytest

from src.assistant.assistant import IngredientAssistant
from src.assistant.intents import classify, rule_intent
from src.assistant.knowledge import knowledge_base

PRODUCT = {
    "entities": [
        {"text": "Sugar", "label": "SUGAR", "start": 0, "end": 5},
        {"text": "Gelatine", "label": "INGREDIENT", "start": 7, "end": 15},
        {"text": "INS 211", "label": "INS_CODE", "start": 31, "end": 38, "ins_number": "211",
         "canonical_name": "Sodium benzoate", "function": "Preservative", "function_source": "label"},
    ],
    "summary": {"groups": {"SUGAR": ["Sugar"], "PRESERVATIVE": ["INS 211"]},
                "hidden": {"sugars_under_other_names": [], "sugar_like_carbohydrates": []}},
    "allergens": {"contains": {"Milk": ["milk solids"]}, "may_contain": {"Tree nuts": ["nuts"]}},
}


@pytest.fixture(scope="module")
def bot():
    return IngredientAssistant(PRODUCT, guide={}, body_weight_kg=60)


# ------------------------------------------------------------------ intents
@pytest.mark.parametrize("question,intent", [
    ("why is citric acid added?", "why_added"), ("is it vegan?", "diet"), ("what allergens are in this", "allergens"),
    ("how much aspartame is safe per day", "daily_limit"), ("which ingredients are sugars", "list_category")])
def test_model_intents(question, intent):
    assert classify(question)[0] == intent


def test_medical_safety_net_overrides_the_model():
    assert classify("I am pregnant, how much sugar is ok?")[0] == "medical"
    assert rule_intent("is this healthy") == "medical"


def test_model_tolerates_typos_the_rules_miss():
    assert classify("alergens?")[0] == "allergens"


# ------------------------------------------------------------------ mentions
def test_mentions_codes_names_and_typos():
    kb = knowledge_base()
    assert [m.key for m in kb.find_mentions("what is E471")] == ["471"]
    assert [m.key for m in kb.find_mentions("why is INS 503(ii) used")] == ["503ii"]
    assert {m.key for m in kb.find_mentions("sodium benzoate and citric acid")} == {"211", "330"}
    fuzzy = kb.find_mentions("what is malodextrin")
    assert fuzzy[0].key == "maltodextrin" and fuzzy[0].method == "fuzzy"


# ------------------------------------------------------------------ answers
def test_answers_come_from_the_knowledge_base_with_sources(bot):
    reply = bot.answer("What is E211?")
    assert "Sodium benzoate" in reply["text"] and "preservative" in reply["text"].lower()
    assert reply["sources"] and not reply["refused"]


def test_it_resolves_to_the_previous_ingredient(bot):
    first = bot.answer("what is E211")
    reply = bot.answer("is it vegetarian?", [first])
    assert reply["resolved_reference"] == "Sodium benzoate"
    assert "vegetarian" in reply["text"]


def test_product_diet_check_finds_animal_ingredients(bot):
    assert "not vegetarian" in bot.answer("is this product vegetarian?")["text"]


def test_contains_and_allergens(bot):
    assert bot.answer("does it contain sodium benzoate?")["text"].startswith("**Yes**")
    text = bot.answer("does it have nuts?")["text"]
    assert "may contain traces" in text


def test_daily_limit_is_scaled_to_body_weight(bot):
    assert "300 mg/day" in bot.answer("what is the ADI of sodium benzoate")["text"]


def test_medical_questions_are_declined_with_facts(bot):
    reply = bot.answer("my child has diabetes, can she eat this?")
    assert reply["refused"] and "medical" in reply["text"]
    assert "Sugar" in reply["text"]


def test_unknown_ingredient_says_so():
    reply = IngredientAssistant().answer("what is zzqqxx")
    assert "couldn't find “zzqqxx”" in reply["text"] and "INS/E code" in reply["text"]


# ------------------------------------------------------------------ out-of-scope questions
@pytest.mark.parametrize("question", ["What is the speed of light?", "what is the capital of france",
                                      "write me a poem", "what is 2+2", "tell me a joke", "how do i fix my laptop"])
def test_off_topic_questions_are_declined_politely(bot, question):
    reply = bot.answer(question)
    assert reply["intent"] == "out_of_scope"
    assert "only answer questions about food ingredients" in reply["text"]


@pytest.mark.parametrize("question", ["is this vegan?", "what's in this?", "what is 330", "hi", "what allergens?",
                                      "any alergens?", "does it have cashews", "is this healthy", "is there gelatine?"])
def test_on_topic_questions_are_not_blocked(bot, question):
    assert bot.answer(question)["intent"] != "out_of_scope"


def test_pronoun_follow_up_stays_in_scope(bot):
    first = bot.answer("what is E211")
    assert bot.answer("why is it added?", [first])["intent"] == "why_added"


# ------------------------------------------------------------------ regulatory status
def test_regulatory_flags_from_the_table():
    kb = knowledge_base()
    assert kb.regulatory_flags("171") == ["Banned in EU"]
    assert kb.regulatory_flags("924") == ["Banned in EU", "Banned in India"]
    assert kb.regulatory_flags("102") == ["Warning label required in EU"]
    assert kb.regulatory_flags("330") == []                     # nothing recorded
    assert kb.find_mentions("is BVO banned")[0].key == "443"   # alias from the table


def test_regulatory_answers_cite_sources_and_dates():
    bot = IngredientAssistant()
    reply = bot.answer("Is titanium dioxide banned?")
    assert reply["intent"] == "regulatory"
    assert "EU: banned" in reply["text"] and "2022/63" in reply["text"] and "Checked" in reply["text"]
    one_region = bot.answer("is red 3 banned in india")["text"]
    assert "India: permitted" in one_region and "US:" not in one_region


def test_regulatory_does_not_overclaim_for_unlisted_additives():
    text = IngredientAssistant().answer("is citric acid banned")["text"]
    assert "no ban" in text and "doesn't prove" in text


def test_product_level_regulatory_check_lists_each_additive_once():
    result = {"entities": [{"text": "Titanium dioxide", "label": "COLOUR", "ins_number": "171", "start": 0, "end": 16},
                           {"text": "171", "label": "INS_CODE", "ins_number": "171", "start": 18, "end": 21},
                           {"text": "102", "label": "INS_CODE", "ins_number": "102", "start": 23, "end": 26}]}
    text = IngredientAssistant(result).answer("is anything in this banned elsewhere?")["text"]
    assert text.count("Titanium dioxide") == 1 and "Tartrazine" in text


def test_regulatory_badges():
    from src.app.ui_components import regulatory_badges
    html = regulatory_badges(["Banned in EU", "Warning label required in EU"])
    assert "Banned in EU" in html and "#C92A2A" in html and "#B76E00" in html
    assert regulatory_badges([]) == ""


# ------------------------------------------------------------------ UI
def test_chat_message_escapes_html_and_renders_bold():
    from src.app.ui_components import chat_message
    html = chat_message("bot", "**Yes** <script>", ["Your scan"])
    assert "<b>Yes</b>" in html and "<script>" not in html and "Your scan" in html
