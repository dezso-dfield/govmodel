from __future__ import annotations

from govmodel.pullers.woo import (
    MACHINE_READABLE_FORMATS,
    filter_machine_readable_resources,
    summarize_dataset,
)


def test_machine_readable_formats_basic_set():
    assert "json" in MACHINE_READABLE_FORMATS
    assert "csv" in MACHINE_READABLE_FORMATS
    assert "pdf" not in MACHINE_READABLE_FORMATS


def test_filter_machine_readable_resources():
    dataset = {
        "resources": [
            {"format": "PDF", "url": "https://x.test/a.pdf"},
            {"format": "JSON", "url": "https://x.test/a.json"},
            {"format": "csv", "url": "https://x.test/a.csv"},
            {"format": "html", "url": "https://x.test/a.html"},
        ]
    }
    machine_readable = filter_machine_readable_resources(dataset)
    formats = {r["format"].lower() for r in machine_readable}
    assert formats == {"json", "csv"}


def test_filter_handles_missing_format():
    dataset = {"resources": [{"url": "https://x.test/y"}, {"format": None, "url": "https://x.test/z"}]}
    assert filter_machine_readable_resources(dataset) == []


def test_filter_handles_no_resources():
    assert filter_machine_readable_resources({}) == []
    assert filter_machine_readable_resources({"resources": None}) == []


def test_summarize_dataset_minimal():
    dataset = {
        "id": "abc",
        "name": "test-dataset",
        "title": "Test WOO",
        "organization": {"title": "Gemeente Testland"},
        "license_id": "cc0-1.0",
        "license_title": "CC0 1.0",
        "metadata_modified": "2024-01-15T10:00:00",
        "notes": "Beschrijving van de dataset.",
        "resources": [
            {"format": "JSON", "url": "https://x.test/a.json", "name": "a"},
            {"format": "PDF", "url": "https://x.test/a.pdf", "name": "b"},
        ],
    }
    s = summarize_dataset(dataset)
    assert s["organization"] == "Gemeente Testland"
    assert s["license_id"] == "cc0-1.0"
    assert s["n_resources_total"] == 2
    assert s["n_resources_machine_readable"] == 1
    assert "json" in s["all_resource_formats"]
    assert "pdf" in s["all_resource_formats"]
    assert s["machine_readable_urls"][0]["format"] == "json"


def test_summarize_dataset_no_organization():
    dataset = {"id": "x", "name": "y", "title": "Z", "resources": []}
    s = summarize_dataset(dataset)
    assert s["organization"] is None
    assert s["n_resources_total"] == 0
