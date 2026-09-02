from collections import namedtuple

from pipeweave_ir.util import canonicalize_text, json_safe, stable_hash


def test_stable_hash_ignores_mapping_order() -> None:
    assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})


def test_canonicalize_text_normalizes_paths_and_whitespace() -> None:
    first = '  %0 = "tt.load"() loc("/tmp/a/kernel.py":7:2)  \r\n'
    second = '  %0 = "tt.load"() loc("/another/path/kernel.py":7:2)\n\n'
    assert canonicalize_text(first) == canonicalize_text(second)


def test_json_safe_preserves_namedtuple_fields() -> None:
    Metadata = namedtuple("Metadata", ["shared", "num_warps"])
    assert json_safe(Metadata(shared=16384, num_warps=4)) == {
        "shared": 16384,
        "num_warps": 4,
    }
