import json

from granite_adapter_guide.evaluation import evaluate_predictions


def write_jsonl(path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def test_evaluates_by_id_with_json_exact_match_and_schema(tmp_path):
    expected = tmp_path / "expected.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    schema = tmp_path / "schema.json"
    write_jsonl(
        expected,
        [
            {"id": "a", "input": "A", "output": '{"answer": 1}'},
            {"id": "b", "input": "B", "output": '{"answer": 2}'},
            {"id": "c", "input": "C", "output": '{"answer": 3}'},
        ],
    )
    write_jsonl(
        predictions,
        [
            {"id": "b", "output": '{"answer": 9}'},
            {"id": "a", "output": '{ "answer" : 1 }'},
            {"id": "extra", "output": '{"answer": 4}'},
            {"id": "c", "output": "not-json"},
        ],
    )
    schema.write_text(
        json.dumps(
            {
                "type": "object",
                "required": ["answer"],
                "properties": {"answer": {"type": "integer"}},
                "additionalProperties": False,
            }
        )
    )

    report = evaluate_predictions(expected, predictions, schema)

    assert report.expected_count == 3
    assert report.prediction_count == 4
    assert report.json_valid_count == 2
    assert report.schema_valid_count == 2
    assert report.exact_match_count == 1
    assert report.missing_ids == []
    assert report.unexpected_ids == ["extra"]
    assert report.exact_match_rate == 1 / 3


def test_missing_predictions_stay_in_denominator_and_line_order_is_supported(tmp_path):
    expected = tmp_path / "expected.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    write_jsonl(expected, [{"input": "A", "output": "1"}, {"input": "B", "output": "2"}])
    write_jsonl(predictions, [{"prediction": "1"}])

    report = evaluate_predictions(expected, predictions)

    assert report.expected_count == 2
    assert report.exact_match_count == 1
    assert report.exact_match_rate == 0.5
    assert report.missing_count == 1
    assert report.schema_valid_count == 1
