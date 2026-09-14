"""Tests for clarity_pii PII detection (detect.py).

Regression coverage for the date-as-phone / date-as-card misclassification:
dates and timestamps are NOT PII and must never be tokenized, while genuine
phone numbers and credit cards must still be detected.
"""
from __future__ import annotations

from pi_coding_agent.clarity_pii.detect import detect


def _types(text: str) -> list[str]:
    return [etype for _, etype in detect(text)]


class TestDatesAreNotPII:
    def test_plain_date_is_not_a_phone(self):
        # The exact production failure: an ISO date was masked as [PII:PHONE:1],
        # corrupting a file path (/tmp/atlas-qual-validation-2026-07-25.md).
        assert "PHONE_NUMBER" not in _types("2026-07-25")

    def test_plain_date_is_not_detected_at_all(self):
        assert detect("2026-07-25") == []

    def test_dash_timestamp_is_not_a_phone(self):
        assert "PHONE_NUMBER" not in _types("2026-07-25-12-04-51")

    def test_dash_timestamp_is_not_a_credit_card(self):
        # 14 dash-separated digits can coincidentally pass Luhn; a date is not
        # a card number.
        assert "CREDIT_CARD" not in _types("2026-07-25-12-04-51")

    def test_dash_timestamp_is_not_detected_at_all(self):
        assert detect("2026-07-25-12-04-51") == []

    def test_iso_t_timestamp_is_not_pii(self):
        assert detect("2026-07-25T12:04:51") == []

    def test_date_inside_a_path_is_left_intact(self):
        assert detect("/tmp/atlas-qual-validation-2026-07-25.md") == []

    def test_date_embedded_in_text_is_not_pii(self):
        assert detect("backup taken 2026-07-25-12-04-51 done") == []


class TestGenuinePhonesStillDetected:
    def test_international_prefix_phone(self):
        assert "PHONE_NUMBER" in _types("call +1 415-555-0132 now")

    def test_parenthesized_area_code_phone(self):
        assert "PHONE_NUMBER" in _types("reach me at (415) 555-0132")

    def test_ten_digit_formatted_phone(self):
        # 10 digits with separators (no country code / parens) → strong phone.
        assert "PHONE_NUMBER" in _types("my cell is 415-555-0132")


class TestAmbiguousNumericDowngrade:
    def test_weak_seven_digit_is_ambiguous_not_phone(self):
        # 7-9 bare digits, no country code / parens → generic, not asserted phone.
        types = _types("ext 555-1234 please")
        assert "PHONE_NUMBER" not in types
        assert "AMBIGUOUS_NUMERIC" in types

    def test_ambiguous_numeric_maps_to_generic_label(self):
        from pi_coding_agent.clarity_pii.detect import label_for

        assert label_for("AMBIGUOUS_NUMERIC") == "REDACTED"


class TestOtherRecognizersUnaffected:
    def test_email_still_detected(self):
        assert "EMAIL_ADDRESS" in _types("write me@example.com anytime")

    def test_ssn_still_detected(self):
        assert "US_SSN" in _types("ssn 123-45-6789")

    def test_real_credit_card_still_detected(self):
        assert "CREDIT_CARD" in _types("card 4111 1111 1111 1111")

    def test_ipv4_still_detected(self):
        assert "IP_ADDRESS" in _types("host 192.168.0.1")


def test_uuid_routing_ids_survive_pii_tokenization_alongside_real_phone():
    from pi_coding_agent.clarity_pii.vault import Vault

    identifier = '0d400000-0000-4000-8000-000000000006'
    text = f'Executor {identifier}; contact +1 250-555-0199; numeric 0000-4000-8000'
    vault = Vault()
    protected = vault.tokenize(text)
    assert identifier in protected
    assert '+1 250-555-0199' not in protected
    assert protected.endswith('[PII:PHONE:2]')
    assert vault.detokenize(protected) == text


def test_uuid_inside_email_remains_pii():
    from pi_coding_agent.clarity_pii.vault import Vault

    identifier = '0d400000-0000-4000-8000-000000000006'
    for email in (identifier + '@example.test', identifier + '+ops@example.test'):
        vault = Vault()
        text = f'Executor {identifier}; email {email}'
        protected = vault.tokenize(text)
        assert email not in protected
        assert identifier in protected
        assert '[PII:EMAIL:' in protected
        assert vault.detokenize(protected) == text


def test_sha256_file_ids_survive_pii_tokenization_with_real_card_protected():
    from pi_coding_agent.clarity_pii.vault import Vault

    digest = '730b45c159508bad86907654977712fadaaaac842ab06271c0f138de384b07dd'
    card = '4111111111111111'
    containing_card = 'a' * 24 + card + 'b' * 24
    text = f'file-{digest}.json hash={containing_card}; card {card}'
    vault = Vault()
    protected = vault.tokenize(text)
    assert digest in protected
    assert containing_card in protected
    assert protected.endswith('[PII:CC:1]')
    assert vault.detokenize(protected) == text


def test_sha256_inside_email_is_still_protected():
    from pi_coding_agent.clarity_pii.vault import Vault

    digest = '730b45c159508bad86907654977712fadaaaac842ab06271c0f138de384b07dd'
    text = f'file {digest}; email {digest}@example.test'
    vault = Vault()
    protected = vault.tokenize(text)
    assert digest in protected
    assert digest + '@example.test' not in protected
    assert '[PII:EMAIL:' in protected
    assert vault.detokenize(protected) == text


def test_log_values_adjacent_to_timestamps_survive_with_real_pii_protected():
    from pi_coding_agent.clarity_pii.vault import Vault

    for separator in ('\n', '\r\n', ' '):
        for status in ('503', '429', '200'):
            evidence = f'deployment_http_status={status}{separator}2026-09-14T12:02:23 INFO worker-143'
            text = evidence + '\nContact +1 (604) 555-0199 or ops@example.test'
            vault = Vault()
            protected = vault.tokenize(text)
            assert evidence in protected
            assert '+1 (604) 555-0199' not in protected
            assert 'ops@example.test' not in protected
            assert vault.detokenize(protected) == text


def test_real_phone_before_timestamp_and_wrapped_phone_remain_protected():
    from pi_coding_agent.clarity_pii.vault import Vault

    for phone in ('+1 604-555-0199', '+1 604\n555-0199'):
        text = phone + '\n2026-09-14T12:02:23 INFO event'
        vault = Vault()
        protected = vault.tokenize(text)
        assert phone not in protected
        assert '2026-09-14T12:02:23' in protected
        assert vault.detokenize(protected) == text


def test_all_dash_timestamp_with_adjacent_status_is_not_phone():
    assert detect("2026-09-14-12-02-23 503") == []
