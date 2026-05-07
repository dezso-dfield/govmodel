from __future__ import annotations

from govmodel.pii import (
    find_pii,
    is_valid_bsn,
    is_valid_iban_nl,
    pseudonymize,
)


# --- BSN -----------------------------------------------------------------


def test_valid_bsn_passes():
    # Een BSN dat de 11-proef passeert (testnummer)
    assert is_valid_bsn("111222333") is True


def test_invalid_bsn_fails_11_proef():
    assert is_valid_bsn("123456789") is False  # voldoet niet aan 11-proef


def test_bsn_wrong_length():
    assert is_valid_bsn("12345678") is False
    assert is_valid_bsn("1234567890") is False


def test_bsn_non_digit():
    assert is_valid_bsn("abcdefghi") is False


# --- IBAN ----------------------------------------------------------------


def test_valid_iban_nl_passes():
    # Officieel ING-testnummer dat de mod-97 passeert
    assert is_valid_iban_nl("NL91ABNA0417164300") is True


def test_iban_with_spaces():
    assert is_valid_iban_nl("NL91 ABNA 0417 1643 00") is True


def test_iban_invalid_checksum():
    assert is_valid_iban_nl("NL00ABNA0417164300") is False


def test_iban_wrong_country():
    assert is_valid_iban_nl("DE91ABNA0417164300") is False


# --- find_pii ------------------------------------------------------------


def test_finds_email():
    text = "Stuur me een bericht op jan.jansen@gemeente-x.nl als je vragen hebt."
    redactions = find_pii(text)
    assert any(r.pattern_name == "EMAIL" for r in redactions)


def test_finds_phone_mobile():
    text = "Bel me op 06-12345678 als het nodig is."
    redactions = find_pii(text)
    assert any(r.pattern_name == "TELEFOON" for r in redactions)


def test_finds_phone_fixed():
    text = "Mijn vaste nummer is 020-1234567."
    redactions = find_pii(text)
    assert any(r.pattern_name == "TELEFOON" for r in redactions)


def test_phone_regex_does_not_match_invalid_area_code():
    """Zaaknummers en dossier-IDs zoals '017185738' mogen niet als telefoon worden gezien
    (017 is geen uitgegeven NL-kengetal)."""
    text = "Zaaknummer 017185738 is niet hetzelfde als een telefoonnummer."
    redactions = find_pii(text)
    phones = [r for r in redactions if r.pattern_name == "TELEFOON"]
    assert phones == []


def test_phone_international():
    text = "Bel +31 6 12345678 of +31206123456."
    redactions = find_pii(text)
    phones = [r for r in redactions if r.pattern_name == "TELEFOON"]
    assert len(phones) >= 1


def test_finds_postcode():
    text = "Mijn adres is Hoofdstraat 12, 1234AB Amsterdam."
    redactions = find_pii(text)
    assert any(r.pattern_name == "POSTCODE" for r in redactions)


def test_finds_postcode_with_space():
    text = "Postcode: 1234 AB"
    redactions = find_pii(text)
    assert any(r.pattern_name == "POSTCODE" for r in redactions)


def test_finds_iban_only_if_valid():
    text_valid = "Mijn rekening: NL91ABNA0417164300"
    text_invalid = "Verzonnen: NL00ABNA0417164300"
    valid_red = find_pii(text_valid)
    invalid_red = find_pii(text_invalid)
    assert any(r.pattern_name == "IBAN" for r in valid_red)
    assert not any(r.pattern_name == "IBAN" for r in invalid_red)


def test_finds_bsn_only_if_valid():
    # 111222333 is geldig (11-proef ok); 111111111 is niet
    text = "Geldig: 111222333. Ongeldig: 111111111."
    redactions = find_pii(text)
    bsn_redactions = [r for r in redactions if r.pattern_name == "BSN"]
    assert len(bsn_redactions) == 1
    assert bsn_redactions[0].original == "111222333"


# --- pseudonymize --------------------------------------------------------


def test_pseudonymize_email_placeholder():
    text = "Mail: piet@voorbeeld.nl"
    redacted, redactions = pseudonymize(text)
    assert "[EMAIL]" in redacted
    assert "piet@voorbeeld.nl" not in redacted
    assert len(redactions) == 1


def test_pseudonymize_multiple_patterns():
    text = (
        "Geachte gemeente, mijn naam is Jan en ik woon op Hoofdstraat 12, "
        "1234AB. Bel me gerust op 06-12345678 of mail jan@example.com. "
        "Mijn rekening is NL91ABNA0417164300."
    )
    redacted, redactions = pseudonymize(text)
    placeholders = {r.placeholder for r in redactions}
    assert "[EMAIL]" in placeholders
    assert "[TELEFOON]" in placeholders
    assert "[POSTCODE]" in placeholders
    assert "[IBAN]" in placeholders


def test_pseudonymize_no_pii():
    text = "Geachte gemeente, ik wil graag een afspraak."
    redacted, redactions = pseudonymize(text)
    assert redacted == text
    assert redactions == []


def test_pseudonymize_preserves_non_pii_text():
    text = "Geachte gemeente, mail mij op jan@example.com voor info."
    redacted, _ = pseudonymize(text)
    assert "Geachte gemeente, mail mij op " in redacted
    assert " voor info." in redacted


def test_pseudonymize_handles_overlap():
    """Geen dubbele redactie als patronen overlappen."""
    # Een telefoonnummer mag niet ook als BSN worden geredacteerd
    text = "Bel me op 06-12345678."
    redacted, redactions = pseudonymize(text)
    # Tellen dat tekst geen onverwachte dubbele [BSN] bevat
    assert redacted.count("[TELEFOON]") == 1
    assert "[BSN]" not in redacted
